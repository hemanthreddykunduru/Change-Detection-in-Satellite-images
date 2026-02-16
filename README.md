# Change Detection Project

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install python-dotenv psycopg2 sahi rasterio opencv-python pillow numpy
   ```

2. **Configure Environment Variables**
   - Copy `.env.example` to `.env`
   - Update the values in `.env` with your actual configuration:
     - Database credentials (user, password, host, port, database name)
     - Folder paths (input folder, detected folder, objects folder)
     - Model path and settings
     - Detection parameters

3. **Run the Application**
   ```bash
   python starter.py
   ```

## Environment Variables

All configuration is stored in the `.env` file:

- **Database**: PostgreSQL connection details
- **Paths**: Input/output folder locations
- **Model**: YOLO model path and settings
- **Detection**: Slice dimensions and overlap ratios

**Note**: The `.env` file is gitignored to protect sensitive credentials. Never commit it to version control.
