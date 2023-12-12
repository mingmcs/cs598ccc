# Steps to run
1. Clone this repo
2. Create/activate a Python 3.10 virtual environment and install dependencies from `requirements.txt`
3. Update the values in `config.toml` as desired
4. Run `main.py`

# Environment Setup for Redis
docker run --name redislocal -p "6379:6379" -d redis/redis-stack:latest
