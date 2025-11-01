-- schema.sql - TrueLand Database Schema with Migration

-- =========================
-- 1. USERS TABLE
-- =========================
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    phone TEXT,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Seed test users
INSERT INTO users (email, password_hash, phone, is_admin)
VALUES 
  ('tester1@gmail.com', 'dummyhash', NULL, FALSE),
  ('tester2@gmail.com', 'dummyhash', NULL, FALSE)
ON CONFLICT (email) DO NOTHING;

-- =========================
-- 2. USER_IMAGE TABLE
-- =========================
CREATE TABLE IF NOT EXISTS user_image (
    id SERIAL PRIMARY KEY,
    image_id TEXT NOT NULL,
    image_role TEXT,
    listing_id INTEGER,  -- FK added later
    asset_id TEXT,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    create_date TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_image_user_id ON user_image(user_id);
CREATE INDEX IF NOT EXISTS idx_user_image_image_id ON user_image(image_id);

-- Seed sample images
INSERT INTO user_image (image_id, image_role, listing_id, asset_id, user_id, create_date)
VALUES 
    ('header', 'header', NULL, '4', (SELECT user_id FROM users WHERE email='tester1@gmail.com'), '2025-01-10'),
    ('header', 'header', NULL, '2', (SELECT user_id FROM users WHERE email='tester2@gmail.com'), '2025-01-10')
ON CONFLICT DO NOTHING;

-- =========================
-- 3. LISTINGS TABLE (with indexes)
-- =========================
CREATE TABLE IF NOT EXISTS listings (
    listing_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    listing_type TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    asset_subtype TEXT NOT NULL,
    asset_legaltype TEXT NOT NULL,
    price NUMERIC(18,2),
    square NUMERIC(10,2),
    province TEXT NOT NULL,
    city TEXT NOT NULL,
    district TEXT NOT NULL,
    main_street TEXT,
    main_street_width NUMERIC(5,2),
    alley BOOLEAN DEFAULT FALSE,
    alley_width NUMERIC(5,2),
    mattien BOOLEAN DEFAULT FALSE,
    google_map_id TEXT,
    thua_dat_id NUMERIC,
    to_ban_do_id NUMERIC,
    user_id INTEGER REFERENCES users(user_id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    status TEXT DEFAULT 'active'
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_listings_listing_type ON listings(listing_type);
CREATE INDEX IF NOT EXISTS idx_listings_asset_type ON listings(asset_type);
CREATE INDEX IF NOT EXISTS idx_listings_asset_subtype ON listings(asset_subtype);
CREATE INDEX IF NOT EXISTS idx_listings_province ON listings(province);
CREATE INDEX IF NOT EXISTS idx_listings_district ON listings(district);
CREATE INDEX IF NOT EXISTS idx_listings_status ON listings(status);
CREATE INDEX IF NOT EXISTS idx_listings_created_at ON listings(created_at);
CREATE INDEX IF NOT EXISTS idx_listings_mattien ON listings(mattien);
CREATE INDEX IF NOT EXISTS idx_listings_alley ON listings(alley);
CREATE INDEX IF NOT EXISTS idx_listings_price ON listings(price);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_listings_type_status ON listings(listing_type, status);
CREATE INDEX IF NOT EXISTS idx_listings_province_district ON listings(province, district);
CREATE INDEX IF NOT EXISTS idx_listings_asset_type_subtype ON listings(asset_type, asset_subtype);

-- Seed sample listings
INSERT INTO listings 
(title, description, listing_type, asset_type, asset_subtype, asset_legaltype, price, square, 
 province, city, district, main_street, main_street_width, alley, alley_width, mattien, 
 google_map_id, thua_dat_id, to_ban_do_id, user_id, status)
VALUES
('Căn hộ Masteri Thảo Điền 2PN',
  'Căn hộ view sông, full nội thất, tầng cao.',
  'Bán', 'Căn hộ', '2PN', 'Sổ hồng', 5800000000, 68.5, 
  'TP.HCM', 'Thành phố Thủ Đức', 'Phường Thảo Điền', 'Nguyễn Văn Hưởng', 
  12.0, FALSE, NULL, TRUE, 'ChIJJ1YqkJQvdTER1sOz_1gMZ3I', 
  1123, 24,
  (SELECT user_id FROM users WHERE email='tester2@gmail.com'),
  'active'),

('Nhà mặt tiền quận 5', 
  'Nhà 3 tầng, vị trí kinh doanh đắc địa, gần chợ Kim Biên.', 
  'Bán', 'Nhà', 'Nhà mặt tiền', 'Hợp đồng sang tên', 11500000000, 
  92.0, 'TP.HCM', 'Quận 5', 'Phường 2', 'Trần Hưng Đạo', 14.0, 
  FALSE, NULL, TRUE, 'ChIJ5aBLplQudTERPwF4lA_vhkk', 221, 18,
  (SELECT user_id FROM users WHERE email='tester1@gmail.com'),
  'active')
ON CONFLICT DO NOTHING;

-- =========================
-- 4. SERVICES TABLE - ✅ UPDATED WITH MIGRATION
-- =========================
CREATE TABLE IF NOT EXISTS services (
    id SERIAL PRIMARY KEY,
    type TEXT NOT NULL,
    title TEXT NOT NULL,           -- ✅ ADDED
    description TEXT,
    provider TEXT,
    contact TEXT,                  -- ✅ ADDED
    listing_id INTEGER REFERENCES listings(listing_id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_services_type ON services(type);
CREATE INDEX IF NOT EXISTS idx_services_listing_id ON services(listing_id);

-- ✅ MIGRATION: Add missing columns if they don't exist
DO $$ 
BEGIN
    -- Add title column if not exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='services' AND column_name='title'
    ) THEN
        ALTER TABLE services ADD COLUMN title TEXT;
        
        -- Set default values for existing records
        UPDATE services SET title = type WHERE title IS NULL;
        
        -- Make it NOT NULL after setting defaults
        ALTER TABLE services ALTER COLUMN title SET NOT NULL;
    END IF;
    
    -- Add contact column if not exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='services' AND column_name='contact'
    ) THEN
        ALTER TABLE services ADD COLUMN contact TEXT;
    END IF;
END $$;

-- Seed sample services
INSERT INTO services (type, title, description, provider, contact)
VALUES
  ('finance', 'Tư vấn vay mua nhà', 'Ước tính hạn mức, lãi suất, phương án trả nợ.', 'REAL Finance Desk', 'tel:+84901234567'),
  ('lawyer', 'Kiểm tra pháp lý BĐS', 'Rà soát hồ sơ, quy hoạch, tranh chấp.', 'REAL Legal Hub', 'mailto:legal@trueland.vn'),
  ('broker', 'Môi giới freelancer', 'Đăng tin, dẫn khách, hỗ trợ công chứng.', 'REAL Broker Network', 'tel:+84907654321')
ON CONFLICT DO NOTHING;

-- =========================
-- 5. ENABLE UNACCENT EXTENSION (for accent-insensitive search)
-- =========================
CREATE EXTENSION IF NOT EXISTS unaccent;

-- =========================
-- 6. ANALYZE TABLES (update statistics)
-- =========================
ANALYZE users;
ANALYZE listings;
ANALYZE services;
ANALYZE user_image;

-- =========================
-- 7. ADMIN USER (optional - for testing)
-- =========================
-- Password: admin123 (hashed with bcrypt)
INSERT INTO users (email, password_hash, is_admin)
VALUES ('admin@trueland.vn', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5TD0YZMQvcu1y', TRUE)
ON CONFLICT (email) DO NOTHING;

-- =========================
-- VERIFICATION QUERIES
-- =========================
-- Check if migration was successful
SELECT 
    table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'public' 
  AND table_name = 'services'
ORDER BY ordinal_position;

-- Count records
SELECT 
    (SELECT COUNT(*) FROM users) as users_count,
    (SELECT COUNT(*) FROM listings) as listings_count,
    (SELECT COUNT(*) FROM services) as services_count,
    (SELECT COUNT(*) FROM user_image) as images_count;