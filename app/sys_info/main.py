from fastapi import FastAPI
import psutil
import platform

app = FastAPI(title="System Information API")

@app.get("/info")
async def get_system_info():
    """
    Returns basic system information including OS, CPU count, and total memory.
    """
    # Fetching system metrics using psutil and platform libraries
    system_info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "cpu_count": psutil.cpu_count(logical=True),
        "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
    }
    
    return system_info

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI application using uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
