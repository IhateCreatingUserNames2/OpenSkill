.PHONY: install test clean 
 
install: 
    uv pip install -e . 
 
test: 
    pytest tests/ 
 
clean: 
    rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ 
