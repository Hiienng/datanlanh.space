Hạ tầng quản lý:

Hạ tầng quản lý (cập nhật):
- GitHub: lưu trữ mã nguồn (frontend & backend)
- Netlify: triển khai frontend, proxy `/api/*` sang backend
- Render: chạy FastAPI backend
- Console Neon: Postgres (managed), dùng `DATABASE_URL` (sslmode=require)
- ImageKit: lưu trữ ảnh/tài liệu; upload an toàn nên thực hiện qua backend (ký/signed) hoặc dùng public key khi demo

Frontend nên gọi API qua `"/api"` (được Netlify proxy sang Render) hoặc cấu hình `window.HOPESEED_API_BASE` nếu muốn chỉ định rõ backend.

Bảng mã màu
1. headline, slogan, subtext #120747
2. logo Gradient #63c8d0 → #77bf65 ≈ Mantis
3. Màu nút HEX #4389bd ≈ Steel Blue
4. Màu nền #bde5fe 


 