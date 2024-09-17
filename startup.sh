#!/bin/bash

# Update package list
sudo apt-get update

# Install dependencies for building SQLite and your app
sudo apt-get install -y build-essential wget libsqlite3-dev

# Download SQLite 3.35.0+ source code
wget https://www.sqlite.org/2021/sqlite-autoconf-3350500.tar.gz

# Extract the downloaded file
tar xvfz sqlite-autoconf-3350500.tar.gz

# Navigate to the extracted directory
cd sqlite-autoconf-3350500

# Build and install SQLite 3.35.0+
./configure
make
sudo make install

# Verify the installation
sqlite3 --version

# Navigate to your app directory (adjust the path as needed)
cd /home/site/wwwroot

# Start your Streamlit app (adjust this command as needed for your setup)
python3 -m streamlit run app.py --server.port $PORT --server.enableCORS false --server.address 0.0.0.0 --server.enableXsrfProtection false
