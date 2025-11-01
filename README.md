# 🏙️ TrueLand – Nền tảng Bất Động Sản Minh Bạch P2P

**TrueLand** là một hệ thống bất động sản **P2P (peer-to-peer)**, nơi **người mua và người bán kết nối trực tiếp**, không cần qua trung gian truyền thống.  
Dự án hướng đến triết lý **R.E.A.L = Reliable • Evident • Authentic • Linked**, xây dựng thị trường minh bạch, giá thật và thông tin thật.

---

## 🌍 Mục tiêu dự án

- Tạo **môi trường giá lành mạnh** dựa trên dữ liệu thị trường thực.
- Cho phép **người bán tự định giá và tự rao bán tài sản**.
- Cung cấp **định giá tham chiếu (market reference)** và theo dõi giá khu vực.
- Mở rộng hệ sinh thái gồm:
  - **Môi giới tự do (freelancer)** chỉ hỗ trợ hồ sơ, giấy tờ khi người bán cần.
  - **AI Legal Advisor / Luật sư freelancer** tư vấn quy trình pháp lý minh bạch.
- Tích hợp dần dữ liệu từ **cơ quan quản lý đất đai** và **hệ thống định danh quốc gia**, hướng đến một nguồn dữ liệu duy nhất “**1 truth of source**”.

---

## ⚙️ Hạ tầng quản lý (updated)

| Thành phần | Vai trò | Ghi chú |
|-------------|----------|---------|
| **GitHub** | Lưu trữ mã nguồn frontend & backend | `findatasolution/trueland` |
| **Render** | Chạy **FastAPI backend** | Lấy biến môi trường `DATABASE_URL` từ Neon |
| **Neon Console** | PostgreSQL managed (cloud) | `sslmode=require`, auto-pooling |
| **ImageKit.io** | Lưu trữ ảnh và tài liệu; upload an toàn qua backend (ký / signed URL) hoặc dùng public key khi demo | CDN tự động |

### 🔗 Kết nối Frontend ↔ Backend

- Frontend nên gọi API qua đường dẫn `/api`  
  <!-- → Netlify tự proxy sang backend Render. (Tạm thời chưa dùng) -->
- Nếu muốn chỉ định rõ backend (trong local hoặc production):  
  ```js
  window.TRUE_LAND_API_BASE = "https://trueland-api.onrender.com"


[Frontend - Netlify]
  |
  |--> /api/*  →  [Backend - FastAPI @ Render]
                        |
                        |--> [Database - Neon PostgreSQL]
                        |
                        |--> [Image Storage - ImageKit.io]
