import pandas as pd
import numpy as np

data = pd.read_csv("clean_data.csv")

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    # Replace invalid values with NaN
    data['ProdID'] = data['ProdID'].replace(-2147483648, np.nan)
    data['ID'] = data['ID'].replace(-2147483648, np.nan)

    # Convert ID to numeric and clean
    data['ID'] = pd.to_numeric(data['ID'], errors='coerce')
    data = data.dropna(subset=['ID'])

    # Clean ProdID
    data = data.dropna(subset=['ProdID'])
    
    # Remove rows where ID or ProdID is 0
    data = data[(data['ID'] != 0) & (data['ProdID'] != 0)].copy()

    data['ID'] = data['ID'].astype('int64')
    data['ProdID'] = data['ProdID'].astype('int64')

    # ReviewCount
    data['ReviewCount'] = pd.to_numeric(
        data['ReviewCount'], errors='coerce'
    ).fillna(0).astype('int64')

    # Drop unwanted column if exists
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])

    # Fill text columns
    for col in ['Category', 'Brand', 'Description', 'Tags']:
        data[col] = data[col].fillna('')

    # ===== CLEAN IMAGE URLs =====
    # Keep only the first image URL, remove everything after "|"
    def clean_image_url(url):
        if pd.isna(url) or url == '':
            return ''
        
        # Convert to string
        url = str(url)
        
        # If "|" exists, take only the first part
        if '|' in url:
            url = url.split('|')[0]
        
        # If comma-separated, take first one
        if ',' in url:
            url = url.split(',')[0]
        
        # Strip whitespace
        url = url.strip()
        
        # Basic validation - must start with http
        if url and not url.startswith('http'):
            return ''
        
        return url
    
    # Apply image URL cleaning
    if 'ImageURL' in data.columns:
        data['ImageURL'] = data['ImageURL'].apply(clean_image_url)
        # Remove rows with empty ImageURL
        data = data[data['ImageURL'] != ''].copy()
    
    # Ensure required columns exist and are not empty
    required_cols = ['Name', 'Brand', 'Rating']
    for col in required_cols:
        if col in data.columns:
            data = data[data[col].notna()].copy()
            if col in ['Name', 'Brand']:
                data = data[data[col] != ''].copy()

    return data