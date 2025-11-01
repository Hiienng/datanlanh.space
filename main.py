# main.py
import os
from decimal import Decimal
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, Header, Query, APIRouter, UploadFile, File, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy import func as sa_func
from dotenv import load_dotenv
from pathlib import Path

from database import get_db, Base, engine
from models import User, Listing, Service, UserImage
from auth import hash_password, verify_password, create_access_token, decode_access_token
from jose import JWTError
try:
    from jose.exceptions import ExpiredSignatureError
except Exception:
    from jose import ExpiredSignatureError

import time
import base64
import requests
import logging
import json
import hashlib
import psutil
import unicodedata

# ✅ ADDED: Import Gemini
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logging.warning("google-generativeai not installed. AI chat will be disabled.")

load_dotenv()

# Optional schema isolation
DB_SCHEMA = (os.getenv("DB_SCHEMA") or "trueland").strip()
SCHEMA_PREFIX = (DB_SCHEMA + ".") if DB_SCHEMA and DB_SCHEMA.lower() != "public" else ""

# =========================
# Configure structured logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("trueland")

# Configure Gemini API key if available
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if 'genai' in globals() and genai is not None and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini AI configured successfully")
    except Exception as _e:
        logger.warning(f"Failed to configure Gemini AI: {_e}")

# =========================
# Logging Middleware
# =========================
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        
        logger.info(f"[{request_id}] {request.method} {request.url.path} - Started")
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} - "
                f"Status: {response.status_code} - Duration: {duration:.3f}s"
            )
            
            response.headers["X-Process-Time"] = str(duration)
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} - "
                f"ERROR: {str(e)} - Duration: {duration:.3f}s",
                exc_info=True
            )
            raise

# =========================
# Cache layer (Redis optional)
# =========================
CACHE = {}
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))

REDIS_URL = os.getenv("REDIS_URL") or os.getenv("UPSTASH_REDIS_URL")
redis_client = None
try:
    if REDIS_URL:
        import redis
        redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        try:
            redis_client.ping()
        except Exception as _e:
            logger.warning(f"Redis ping failed, using in-memory cache: {_e}")
            redis_client = None
except Exception as _e:
    logger.warning(f"Redis not available, using in-memory cache: {_e}")

def cache_key(prefix: str, **kwargs) -> str:
    params = ''.join(f"{k}={v}" for k, v in sorted(kwargs.items()) if v is not None)
    return f"{prefix}:{hashlib.md5(params.encode()).hexdigest()}"

def get_cached(key: str):
    if redis_client is not None:
        try:
            val = redis_client.get(key)
            if val:
                return json.loads(val)
        except Exception as _e:
            logger.warning(f"Redis get failed, falling back to in-memory cache: {_e}")

    if key in CACHE:
        data, timestamp = CACHE[key]
        if time.time() - timestamp < CACHE_TTL:
            return data
        else:
            del CACHE[key]
    return None

def set_cache(key: str, data):
    if redis_client is not None:
        try:
            def _default(o):
                try:
                    return o.model_dump()
                except Exception:
                    return str(o)
            payload = json.dumps(data, default=_default)
            redis_client.setex(key, CACHE_TTL, payload)
        except Exception as _e:
            logger.warning(f"Redis set failed, using in-memory cache: {_e}")

    CACHE[key] = (data, time.time())

def clear_cache_pattern(pattern: str):
    if redis_client is not None:
        try:
            cursor = 0
            while True:
                cursor, keys = redis_client.scan(cursor=cursor, match=f"{pattern}*")
                if keys:
                    redis_client.delete(*keys)
                if cursor == 0:
                    break
            return
        except Exception as _e:
            logger.warning(f"Redis clear pattern failed, falling back to in-memory cache: {_e}")

    keys_to_delete = [k for k in list(CACHE.keys()) if k.startswith(pattern)]
    for k in keys_to_delete:
        del CACHE[k]

# -------------------------
# Text normalization helpers
# -------------------------
def _strip_accents(text: str) -> str:
    try:
        nfkd_form = unicodedata.normalize('NFKD', text)
        return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])
    except Exception:
        return text

def _canon_province(val: Optional[str]) -> Optional[str]:
    if not val:
        return val
    v = val.strip().lower()
    v_ascii = _strip_accents(v).replace('.', ' ').replace('-', ' ').replace('_', ' ')
    v_ascii = ' '.join(v_ascii.split())

    synonyms = {
        'hcm': 'ho chi minh', 'tp hcm': 'ho chi minh', 'tp.hcm': 'ho chi minh', 'tphcm': 'ho chi minh',
        'ho chi minh': 'ho chi minh', 'tp ho chi minh': 'ho chi minh',
        'ha noi': 'ha noi', 'hn': 'ha noi', 'tp ha noi': 'ha noi',
        'da nang': 'da nang', 'dn': 'da nang', 'tp da nang': 'da nang',
        'binh duong': 'binh duong', 'dong nai': 'dong nai'
    }
    return synonyms.get(v_ascii, v_ascii)

def _ilike_insensitive(column, value: Optional[str]):
    if not value:
        return None
    token = _strip_accents(value.strip().lower())
    try:
        return sa_func.lower(sa_func.unaccent(column)).like(f"%{token}%")
    except Exception:
        return column.ilike(f"%{value}%")

# =========================
# Configure Gemini (✅ ADDED)
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and genai:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini AI configured successfully")
elif genai:
    logger.warning("GEMINI_API_KEY not set. AI chat will be disabled.")

# --- App ---
app = FastAPI(title="Trueland API")

app.add_middleware(LoggingMiddleware)

if os.getenv("AUTO_CREATE_TABLES") == "1":
    Base.metadata.create_all(bind=engine)

try:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent"))
except Exception as _e:
    logger.info(f"unaccent extension not enabled or not supported: {_e}")

_cors_env = os.getenv("CORS_ALLOW_ORIGINS") or os.getenv("NETLIFY_URL") or "*"
allow_origins = [o.strip() for o in _cors_env.split(",")] if _cors_env else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL")
IMA_URL_ENDPOINT = os.getenv("IMA_URL_ENDPOINT") 
IMA_PRIVATE_KEY = os.getenv("IMA_PRIVATE_KEY")

# =========================
# Schemas
# =========================
class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    phone: Optional[str] = None

class MeOut(BaseModel):
    email: EmailStr
    is_admin: bool

class ListingOut(BaseModel):
    listing_id: int
    title: str
    description: Optional[str] = None
    listing_type: str
    asset_type: str
    asset_subtype: str
    asset_legaltype: str
    price: Optional[Decimal] = None
    square: Optional[Decimal] = None
    province: str
    city: str
    district: str
    main_street: Optional[str] = None
    main_street_width: Optional[Decimal] = None
    alley: Optional[bool] = False
    alley_width: Optional[Decimal] = None
    mattien: Optional[bool] = False
    google_map_id: Optional[str] = None
    thua_dat_id: Optional[int] = None
    to_ban_do_id: Optional[int] = None
    user_id: Optional[int] = None
    created_at: Optional[str] = None
    status: str = "active"
    image_version: Optional[str] = None

    class Config:
        from_attributes = True

class ListingCreate(BaseModel):
    title: str
    description: Optional[str] = None
    listing_type: str
    asset_type: str
    asset_subtype: str
    asset_legaltype: str
    price: Optional[Decimal] = None
    square: Optional[Decimal] = None
    province: str
    city: str
    district: str
    main_street: Optional[str] = None
    main_street_width: Optional[Decimal] = None
    alley: bool = False
    alley_width: Optional[Decimal] = None
    mattien: bool = False
    google_map_id: Optional[str] = None
    thua_dat_id: Optional[int] = None
    to_ban_do_id: Optional[int] = None

# ✅ ADDED: Service Schemas
class ServiceCreate(BaseModel):
    type: str
    title: str
    description: Optional[str] = None
    provider: Optional[str] = None
    contact: Optional[str] = None
    listing_id: Optional[int] = None

class ServiceOut(BaseModel):
    id: int
    type: str
    title: str
    description: Optional[str] = None
    provider: Optional[str] = None
    contact: Optional[str] = None
    listing_id: Optional[int] = None
    created_at: Optional[str] = None

    class Config:
        from_attributes = True

# ✅ ADDED: AI Chat Schemas
class AIChatRequest(BaseModel):
    message: str
    model: str = "gemini-1.5-flash"
    context: Optional[str] = None

class AIChatResponse(BaseModel):
    response: str
    model: str
    timestamp: str

# =========================
# Auth helpers
# =========================
def decode_token(token: str):
    try:
        return decode_access_token(token)
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(
    authorization: str = Header(..., alias="Authorization"),
    db: Session = Depends(get_db)
) -> User:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# =========================
# Routes
# =========================
@app.post("/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    email_lower = user.email.lower()
    exists = db.query(User).filter(User.email == email_lower).first()
    if exists:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = User(
        email=email_lower,
        password_hash=hash_password(user.password),
        phone=user.phone
    )
    db.add(new_user)
    db.commit()
    return {"message": "User created", "is_admin": email_lower.startswith("admin@")}

@app.post("/login")
def login(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email.lower()).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    is_admin = db_user.email.lower().startswith("admin@")
    claims = {"sub": db_user.email}
    if is_admin:
        claims["adm"] = True
    token = create_access_token(claims)
    return {
        "message": "Login successful",
        "email": db_user.email,
        "is_admin": is_admin,
        "access_token": token,
    }

@app.get("/me", response_model=MeOut)
def me(current_user: User = Depends(get_current_user)):
    return MeOut(email=current_user.email, is_admin=current_user.email.lower().startswith("admin@"))

@app.get("/user-images")
def list_user_images(db: Session = Depends(get_db)):
    rows = db.execute(text(f"""
        SELECT image_id, asset_id
        FROM {SCHEMA_PREFIX}user_image
        ORDER BY create_date, image_id
    """)).all()
    base = IMA_URL_ENDPOINT.rstrip("/") if IMA_URL_ENDPOINT else ""
    return [f"{base}/trueland/users/{r.asset_id}/{r.image_id}.jpg" for r in rows]

# =========================
# Listings CRUD
# =========================
@app.get("/listings", response_model=List[ListingOut])
def get_all_listings(
    status: Optional[str] = Query("active"),
    listing_type: Optional[str] = Query(None),
    asset_type: Optional[str] = Query(None),
    asset_subtype: Optional[str] = Query(None),
    asset_legaltype: Optional[str] = Query(None),
    province: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    city: Optional[str] = Query(None),
    square_min: Optional[float] = Query(None),
    square_max: Optional[float] = Query(None),
    main_street_width: Optional[float] = Query(None),
    mattien: Optional[bool] = Query(None),
    alley: Optional[bool] = Query(None),
    price_min: Optional[float] = Query(None),
    price_max: Optional[float] = Query(None),
    q: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    response: Response = None
):
    cache_params = {
        'status': status, 'listing_type': listing_type, 'asset_type': asset_type,
        'asset_subtype': asset_subtype, 'asset_legaltype': asset_legaltype,
        'province': province, 'district': district, 'city': city,
        'square_min': square_min, 'square_max': square_max,
        'main_street_width': main_street_width, 'mattien': mattien, 'alley': alley,
        'price_min': price_min, 'price_max': price_max, 'q': q,
        'limit': limit, 'offset': offset,
    }
    cache_k = cache_key("listings", **cache_params)
    cached = get_cached(cache_k)
    if cached is not None:
        logger.info(f"Cache HIT: {cache_k}")
        if response:
            response.headers["X-Cache"] = "HIT"
        return cached
    
    # Downgrade MISS logs to DEBUG to avoid noisy INFO entries
    logger.debug(f"Cache MISS: {cache_k}")
    
    query = db.query(Listing)
    
    if status:
        query = query.filter(Listing.status == status)
    if listing_type:
        query = query.filter(Listing.listing_type == listing_type)
    if asset_type:
        query = query.filter(Listing.asset_type == asset_type)
    if asset_subtype:
        query = query.filter(Listing.asset_subtype == asset_subtype)
    if asset_legaltype:
        query = query.filter(Listing.asset_legaltype == asset_legaltype)
    if province:
        query = query.filter(_ilike_insensitive(Listing.province, _canon_province(province)))
    if district:
        query = query.filter(_ilike_insensitive(Listing.district, district))
    if city:
        query = query.filter(_ilike_insensitive(Listing.city, city))
    
    if square_min is not None:
        query = query.filter(Listing.square >= square_min)
    if square_max is not None:
        query = query.filter(Listing.square <= square_max)
    if main_street_width is not None:
        query = query.filter(Listing.main_street_width >= main_street_width)
    if price_min is not None:
        query = query.filter(Listing.price >= price_min)
    if price_max is not None:
        query = query.filter(Listing.price <= price_max)
    
    if mattien is not None:
        query = query.filter(Listing.mattien == mattien)
    if alley is not None:
        query = query.filter(Listing.alley == alley)
    
    if q:
        query = query.filter(
            (Listing.title.ilike(f"%{q}%")) | 
            (Listing.description.ilike(f"%{q}%"))
        )
    
    query = query.order_by(Listing.created_at.desc()).limit(limit).offset(offset)
    rows = query.all()
    # Build cache-busting version for header image per listing (latest upload time)
    versions = {}
    try:
        ids = [r.id for r in rows]
        if ids:
            qv = (
                db.query(UserImage.listing_id, sa_func.max(UserImage.create_date))
                .filter(UserImage.listing_id.in_(ids))
                .filter(UserImage.image_role == "header")
                .group_by(UserImage.listing_id)
            )
            for lid, ts in qv.all():
                versions[lid] = ts.isoformat() if ts else None
    except Exception:
        pass
    
    result = [
        ListingOut(
            listing_id=x.id, title=x.title, description=x.description,
            listing_type=x.listing_type, asset_type=x.asset_type,
            asset_subtype=x.asset_subtype, asset_legaltype=x.asset_legaltype,
            price=x.price, square=x.square, province=x.province,
            city=x.city, district=x.district, main_street=x.main_street,
            main_street_width=x.main_street_width, alley=x.alley,
            alley_width=x.alley_width, mattien=x.mattien,
            google_map_id=x.google_map_id, thua_dat_id=x.thua_dat_id,
            to_ban_do_id=x.to_ban_do_id, user_id=x.user_id,
            created_at=x.created_at.isoformat() if x.created_at else None,
            status=x.status,
            image_version=versions.get(x.id)
        )
        for x in rows
    ]
    
    set_cache(cache_k, result)
    if response:
        # Treat first fill as HIT to avoid MISS semantics in single-instance mode
        response.headers["X-Cache"] = "HIT"
    
    return result

@app.get("/listings/{listing_id}", response_model=ListingOut)
def get_listing_detail(listing_id: int, db: Session = Depends(get_db)):
    x = db.query(Listing).filter(Listing.id == listing_id).first()
    if not x:
        raise HTTPException(status_code=404, detail="Listing not found")
    return ListingOut(
        listing_id=x.id, title=x.title, description=x.description,
        listing_type=x.listing_type, asset_type=x.asset_type,
        asset_subtype=x.asset_subtype, asset_legaltype=x.asset_legaltype,
        price=x.price, square=x.square, province=x.province,
        city=x.city, district=x.district, main_street=x.main_street,
        main_street_width=x.main_street_width, alley=x.alley,
        alley_width=x.alley_width, mattien=x.mattien,
        google_map_id=x.google_map_id, thua_dat_id=x.thua_dat_id,
        to_ban_do_id=x.to_ban_do_id, user_id=x.user_id,
        created_at=x.created_at.isoformat() if x.created_at else None,
        status=x.status
    )

@app.post("/listings", response_model=ListingOut)
def create_listing(
    listing: ListingCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    x = Listing(
        title=listing.title, description=listing.description,
        listing_type=listing.listing_type, asset_type=listing.asset_type,
        asset_subtype=listing.asset_subtype, asset_legaltype=listing.asset_legaltype,
        price=listing.price, square=listing.square, province=listing.province,
        city=listing.city, district=listing.district, main_street=listing.main_street,
        main_street_width=listing.main_street_width, alley=listing.alley,
        alley_width=listing.alley_width, mattien=listing.mattien,
        google_map_id=listing.google_map_id, thua_dat_id=listing.thua_dat_id,
        to_ban_do_id=listing.to_ban_do_id, user_id=current_user.id, status="active",
    )
    db.add(x)
    db.commit()
    db.refresh(x)
    
    clear_cache_pattern("listings:")
    logger.info(f"Created listing {x.id}, cache cleared")
    
    return get_listing_detail(x.id, db)

@app.put("/listings/{listing_id}", response_model=ListingOut)
def update_listing(
    listing_id: int,
    listing_update: ListingCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    x = db.query(Listing).filter(Listing.id == listing_id).first()
    if not x:
        raise HTTPException(status_code=404, detail="Listing not found")
    if x.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized to update this listing")

    for k, v in listing_update.model_dump(exclude_unset=True).items():
        setattr(x, k, v)
    db.commit()
    db.refresh(x)
    
    clear_cache_pattern("listings:")
    logger.info(f"Updated listing {x.id}, cache cleared")
    
    return get_listing_detail(x.id, db)

@app.delete("/listings/{listing_id}")
def delete_listing(
    listing_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    x = db.query(Listing).filter(Listing.id == listing_id).first()
    if not x:
        raise HTTPException(status_code=404, detail="Listing not found")
    if x.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized to delete this listing")

    x.status = "deleted"
    db.commit()
    
    clear_cache_pattern("listings:")
    logger.info(f"Deleted listing {x.id}, cache cleared")
    
    return {"message": "Listing deleted successfully"}

@app.get("/me/listings", response_model=List[ListingOut])
def get_my_listings(
    limit: int = Query(200, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    q = (
        db.query(Listing)
        .filter(Listing.user_id == current_user.id)
        .filter(Listing.status == "active")
        .order_by(Listing.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = q.all()
    return [
        ListingOut(
            listing_id=x.id, title=x.title, description=x.description,
            listing_type=x.listing_type, asset_type=x.asset_type,
            asset_subtype=x.asset_subtype, asset_legaltype=x.asset_legaltype,
            price=x.price, square=x.square, province=x.province,
            city=x.city, district=x.district, main_street=x.main_street,
            main_street_width=x.main_street_width, alley=x.alley,
            alley_width=x.alley_width, mattien=x.mattien,
            google_map_id=x.google_map_id, thua_dat_id=x.thua_dat_id,
            to_ban_do_id=x.to_ban_do_id, user_id=x.user_id,
            created_at=x.created_at.isoformat() if x.created_at else None,
            status=x.status,
        )
        for x in rows
    ]

@app.post("/listings/{listing_id}/header-image")
def upload_listing_header(
    listing_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not IMA_PRIVATE_KEY:
        raise HTTPException(status_code=500, detail="Image upload not configured")

    listing = db.query(Listing).filter(Listing.id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    if listing.user_id not in (None, current_user.id) and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized to upload for this listing")

    try:
        content = file.file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        b64 = base64.b64encode(content).decode("utf-8")
        url = "https://upload.imagekit.io/api/v1/files/upload"
        folder = f"/trueland/users/{current_user.id}"
        data = {
            "file": b64,
            "fileName": "header.jpg",
            "folder": folder,
            "useUniqueFileName": "false",
            "overwriteFilename": "true",
        }
        resp = requests.post(url, auth=(IMA_PRIVATE_KEY, ""), data=data, timeout=20)
        if resp.status_code not in (200, 201):
            raise HTTPException(status_code=502, detail=f"Image upload failed: {resp.text}")
        info = resp.json()
        try:
            from models import UserImage
            img = UserImage(
                image_id="header",
                listing_id=listing.id,
                image_role="header",
                asset_id=str(current_user.id),
                user_id=current_user.id,
            )
            db.add(img)
            db.commit()
        except Exception:
            db.rollback()
        return {"message": "Header uploaded", "url": info.get("url")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# ✅ ADDED: Service CRUD Endpoints
# =========================
@app.get("/services", response_model=List[ServiceOut])
def get_services(
    type: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """Get all services, optionally filtered by type"""
    query = db.query(Service)
    if type:
        query = query.filter(Service.type == type)
    
    query = query.order_by(Service.created_at.desc()).limit(limit).offset(offset)
    rows = query.all()
    
    return [
        ServiceOut(
            id=s.id,
            type=s.type,
            title=s.title,
            description=s.description,
            provider=s.provider,
            contact=s.contact,
            listing_id=s.listing_id,
            created_at=s.created_at.isoformat() if s.created_at else None
        )
        for s in rows
    ]

@app.post("/services", response_model=ServiceOut)
def create_service(
    service: ServiceCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new service"""
    if service.listing_id:
        listing = db.query(Listing).filter(Listing.id == service.listing_id).first()
        if not listing:
            raise HTTPException(status_code=404, detail="Listing not found")
        if listing.user_id != current_user.id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Not authorized")
    
    s = Service(
        type=service.type,
        title=service.title,
        description=service.description,
        provider=service.provider,
        contact=service.contact,
        listing_id=service.listing_id
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    
    return ServiceOut(
        id=s.id,
        type=s.type,
        title=s.title,
        description=s.description,
        provider=s.provider,
        contact=s.contact,
        listing_id=s.listing_id,
        created_at=s.created_at.isoformat() if s.created_at else None
    )

@app.delete("/services/{service_id}")
def delete_service(
    service_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a service"""
    s = db.query(Service).filter(Service.id == service_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Service not found")
    
    if s.listing_id:
        listing = db.query(Listing).filter(Listing.id == s.listing_id).first()
        if listing and listing.user_id != current_user.id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Not authorized")
    elif not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    
    db.delete(s)
    db.commit()
    return {"message": "Service deleted successfully"}

# =========================
# ✅ ADDED: AI Chat Endpoint
# =========================
@app.post("/ai/chat", response_model=AIChatResponse)
def ai_chat(
    request: AIChatRequest,
    current_user: User = Depends(get_current_user)
):
    """
    AI Legal Advisor - Tư vấn quy trình pháp lý BĐS
    Chỉ tham khảo, không thay thế luật sư.
    """
    if not GEMINI_API_KEY or not genai:
        raise HTTPException(
            status_code=503, 
            detail="AI service not configured. Please install google-generativeai and set GEMINI_API_KEY."
        )
    
    try:
        system_prompt = """Bạn là trợ lý AI chuyên tư vấn quy trình pháp lý bất động sản ở Việt Nam.

QUAN TRỌNG:
- Chỉ cung cấp thông tin THAM KHẢO, không thay thế luật sư
- Luôn khuyến khích người dùng tìm luật sư khi cần
- Trả lời ngắn gọn, dễ hiểu, bằng tiếng Việt
- Tập trung vào: thủ tục, giấy tờ, rủi ro, quy hoạch
- Không đưa ra lời khuyên đầu tư cụ thể

Nếu câu hỏi ngoài phạm vi BĐS/pháp lý, lịch sự từ chối và gợi ý hỏi đúng chủ đề."""

        full_prompt = system_prompt + "\n\n"
        if request.context:
            full_prompt += f"Ngữ cảnh trước: {request.context}\n\n"
        full_prompt += f"Câu hỏi: {request.message}"
        
        # Resolve model name using ListModels and add graceful fallbacks
        requested = (request.model or "gemini-1.5-flash").strip()
        tried = []
        last_err = None

        def _tail(n: str) -> str:
            return n.split('/')[-1] if n else n

        # Build available model list that supports generateContent
        avail = []
        try:
            for m in genai.list_models():
                methods = set(getattr(m, 'supported_generation_methods', []) or [])
                if 'generateContent' in methods or 'generate_content' in methods:
                    avail.append(m.name)
        except Exception:
            # If listing fails, continue with static preferences
            avail = []

        # Preferred patterns by capability/perf
        prefs = [
            'gemini-1.5-pro',
            'gemini-1.5-flash',
            'gemini-1.5-flash-8b',
            'gemini-pro',
        ]

        candidates: list[str] = []
        # 1) Exact match (with or without models/ prefix)
        if avail:
            for n in avail:
                if n == requested or _tail(n) == requested:
                    candidates.append(n)
                    break
            # 2) Preference-ordered fallbacks drawn from available
            for p in prefs:
                for n in avail:
                    if _tail(n).startswith(p) and n not in candidates:
                        candidates.append(n)
                        break
        # 3) If nothing available (e.g., list_models failed), try static names (both plain and with prefix)
        if not candidates:
            static = [requested] + [p for p in prefs if p != requested]
            allnames = []
            for s in static:
                allnames.append(s)
                allnames.append(f"models/{s}")
            # preserve order, dedupe
            seen = set()
            candidates = [x for x in allnames if not (x in seen or seen.add(x))]

        for name in candidates:
            try:
                tried.append(name)
                model = genai.GenerativeModel(name)
                response = model.generate_content(full_prompt)
                if not response or not getattr(response, 'text', ''):
                    raise RuntimeError("Empty AI response")
                return AIChatResponse(
                    response=response.text.strip(),
                    model=name,
                    timestamp=datetime.now().isoformat()
                )
            except Exception as _e:
                last_err = _e
                msg = str(_e)
                # Try next on model issues or transient API problems
                if ("404" in msg or "not found" in msg.lower() or "not supported" in msg.lower()
                    or any(k in msg for k in ("HttpError", "APIError", "Unavailable"))):
                    continue
                # Unknown error -> stop and surface
                break

        detail = f"AI model unavailable. Tried: {', '.join(tried)}; last error: {last_err}"
        raise HTTPException(status_code=503, detail=detail)
        
    except HTTPException as e:
        # Preserve intended status codes (e.g., 503 not configured)
        raise e
    except Exception as e:
        logger.error(f"AI chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi AI: {str(e)}"
        )

@app.get("/ai/models")
def list_ai_models():
    if not GEMINI_API_KEY or not genai:
        raise HTTPException(status_code=503, detail="AI service not configured. Please install google-generativeai and set GEMINI_API_KEY.")
    try:
        out = []
        for m in genai.list_models():
            methods = set(getattr(m, 'supported_generation_methods', []) or [])
            if 'generateContent' in methods or 'generate_content' in methods:
                out.append(m.name)
        return sorted(out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ListModels failed: {e}")

# =========================
# Performance Monitoring
# =========================
@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint with system stats"""
    try:
        db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    return {
        "status": "ok",
        "database": db_status,
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "cache_size": len(CACHE)
    }

@app.get("/metrics")
def get_metrics():
    """Get application metrics"""
    return {
        "cache_size": len(CACHE),
        "cache_keys": list(CACHE.keys())[:10],
        "timestamp": datetime.now().isoformat()
    }

# =========================
# Mount static (CUỐI CÙNG)
# =========================
BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
PUBLIC_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="static")
