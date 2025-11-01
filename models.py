# models.py
from sqlalchemy import (
    Column, Integer, String, Boolean, ForeignKey, DateTime, func,
    Numeric, Text, Index
)
from sqlalchemy.orm import relationship
from database import Base


# ======================
# Users
# ======================
class User(Base):
    __tablename__ = "users"

    id = Column("user_id", Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    phone = Column(String, nullable=True)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Quan hệ
    images = relationship(
        "UserImage",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    listings = relationship(
        "Listing",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r}>"


# ======================
# User Images (gallery)
# ======================
class UserImage(Base):
    __tablename__ = "user_image"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(String, index=True, nullable=False)
    listing_id = Column(Integer, ForeignKey("listings.listing_id", ondelete="CASCADE"), nullable=True)
    image_role = Column(String, nullable=True)
    asset_id = Column(String, nullable=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    create_date = Column(DateTime, server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="images")

    def __repr__(self) -> str:
        return f"<UserImage id={self.id} image_id={self.image_id!r} user_id={self.user_id}>"


# ======================
# Listings
# ======================
class Listing(Base):
    __tablename__ = "listings"
    
    id = Column(Integer, primary_key=True, index=True, name="listing_id")
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    listing_type = Column(Text, nullable=False, index=True)
    asset_type = Column(Text, nullable=False, index=True)
    asset_subtype = Column(Text, nullable=False, index=True)
    asset_legaltype = Column(Text, nullable=False)
    price = Column(Numeric(18, 2), nullable=True, index=True)
    square = Column(Numeric(10, 2), nullable=True)
    province = Column(Text, nullable=False, index=True)
    city = Column(Text, nullable=False)
    district = Column(Text, nullable=False, index=True)
    main_street = Column(Text, nullable=True)
    main_street_width = Column(Numeric(5, 2), nullable=True)
    alley = Column(Boolean, default=False, nullable=False, index=True)
    alley_width = Column(Numeric(5, 2), nullable=True)
    mattien = Column(Boolean, default=False, nullable=False, index=True)
    google_map_id = Column(Text, nullable=True)
    thua_dat_id = Column(Numeric, nullable=True)
    to_ban_do_id = Column(Numeric, nullable=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    status = Column(Text, default="active", nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="listings")
    services = relationship("Service", back_populates="listing", cascade="all, delete-orphan", passive_deletes=True)

    # Composite indexes for common filter combinations
    __table_args__ = (
        Index('idx_listing_type_status', 'listing_type', 'status'),
        Index('idx_province_district', 'province', 'district'),
        Index('idx_asset_type_subtype', 'asset_type', 'asset_subtype'),
    )


# ======================
# Services (finance/lawyer/broker) - ✅ UPDATED
# ======================
class Service(Base):
    __tablename__ = "services"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String, nullable=False)        # "finance" / "lawyer" / "broker"
    title = Column(String, nullable=False)       # ✅ ADDED
    description = Column(String, nullable=True)
    provider = Column(String, nullable=True)
    contact = Column(String, nullable=True)      # ✅ ADDED

    listing_id = Column(
        Integer,
        ForeignKey("listings.listing_id", ondelete="CASCADE"),
        nullable=True
    )
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    listing = relationship("Listing", back_populates="services")

    def __repr__(self) -> str:
        return f"<Service id={self.id} type={self.type!r} title={self.title!r}>"