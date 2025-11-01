# # ==========================================
# # 7. PERFORMANCE MONITORING (monitor.py)
# # ==========================================
# import psutil
# from fastapi import APIRouter

# router = APIRouter()

# @router.get("/health")
# def health_check(db: Session = Depends(get_db)):
#     """Health check endpoint with system stats"""
#     try:
#         # Check database
#         db.execute(text("SELECT 1"))
#         db_status = "healthy"
#     except Exception as e:
#         logger.error(f"Database health check failed: {e}")
#         db_status = "unhealthy"
    
#     return {
#         "status": "ok",
#         "database": db_status,
#         "cpu_percent": psutil.cpu_percent(),
#         "memory_percent": psutil.virtual_memory().percent,
#         "cache_size": len(CACHE)
#     }

# @router.get("/metrics")
# def get_metrics():
#     """Get application metrics"""
#     return {
#         "cache_size": len(CACHE),
#         "cache_keys": list(CACHE.keys())[:10],  # Sample of cache keys
#         "timestamp": datetime.now().isoformat()
#     }