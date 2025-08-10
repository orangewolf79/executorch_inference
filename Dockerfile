FROM ubuntu:latest

RUN apt-get update && apt-get install -y build-essential cmake python3 python3-pip python3-venv git curl zstd     && rm -rf /var/lib/apt/lists/*

# Download and install buck2 - CURRENTLY CAUSING ERRORS
RUN curl -L -o buck2-x86_64-unknown-linux-gnu.zst https://github.com/facebook/buck2/releases/latest/download/buck2-x86_64-unknown-linux-gnu.zst
RUN zstd -d buck2-x86_64-unknown-linux-gnu.zst -o /usr/local/bin/buck2
RUN chmod +x /usr/local/bin/buck2

WORKDIR /app

COPY . /app

# Install Python dependencies
RUN python3 -m venv /venv
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

# Create build dir, configure and build the project
RUN mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release
# CURRENTLY CAUSING ERRORS
RUN cmake --build .

# run executable
CMD ["./build/main"]
