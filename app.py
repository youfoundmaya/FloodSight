import streamlit as st
import pickle
import xgboost
import requests
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import st_folium
import numpy as np
import random

# Page configuration
st.set_page_config(
    page_title="FloodSight - Route Analysis & Mapping",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load flood data
@st.cache_data
def load_flood_data():
    try:
        df = pd.read_csv('thane_flood_data.csv', encoding='latin-1')

        df.columns = df.columns.str.strip()
        
        # Fill NaN values in risk columns
        risk_columns = ['FloodRiskLevel', 'RoadRiskLevel', 'HomeRiskLevel']
        for col in risk_columns:
            if col in df.columns:
                df[col] = df[col].fillna('None').astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error loading flood data: {e}")
        return None

def get_coordinates(location):
    """Get latitude and longitude for a location using OpenStreetMap Nominatim API"""
    try:
        url = f"https://nominatim.openstreetmap.org/search"
        params = {
            'q': location,
            'format': 'json',
            'limit': 1
        }
        headers = {'User-Agent': 'FloodSight/1.0'}
        
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return lat, lon
        else:
            raise Exception(f"Location '{location}' not found")
            
    except Exception as e:
        raise Exception(f"Error getting coordinates for {location}: {str(e)}")

def get_weather_data(lat, lon):
    """Get current weather data using OpenWeatherMap API"""
    try:
        # You'll need to get a free API key from openweathermap.org
        API_KEY = "youropenweathermap_api_key"  # Replace with your actual API key
        
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': API_KEY,
            'units': 'metric'  # For Celsius and m/s
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code == 200:
            weather = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind'].get('speed', 0),
                'rainfall': data.get('rain', {}).get('1h', 0)  # mm in last hour
            }
            return weather
        else:
            raise Exception(f"Weather API error: {data.get('message', 'Unknown error')}")
            
    except Exception as e:
        # Fallback to realistic dummy data if API fails
        st.warning(f"Using fallback weather data: {str(e)}")
        return {
            'temperature': 25.0,
            'humidity': 65.0,
            'pressure': 1013.25,
            'wind_speed': 5.0,
            'rainfall': 0.0
        }

def get_tide_level(lat, lon):
    """Get tide level data (simplified - you might need a specialized tide API)"""
    try:
        import math
        current_hour = datetime.now().hour
        tide_factor = math.sin((current_hour / 12) * math.pi)
        coastal_factor = 1.0 if abs(lat) < 60 else 0.5
        tide_level = tide_factor * 1.5 * coastal_factor
        return tide_level
    except Exception:
        return 0.0

def generate_features_from_locations(start_location, end_location):
    """Generate the 6 weather features your model expects"""
    try:
        # Get coordinates for both locations
        start_lat, start_lon = get_coordinates(start_location)
        end_lat, end_lon = get_coordinates(end_location)
        
        # Get weather data for both locations
        start_weather = get_weather_data(start_lat, start_lon)
        end_weather = get_weather_data(end_lat, end_lon)
        
        # Get tide levels
        start_tide = get_tide_level(start_lat, start_lon)
        end_tide = get_tide_level(end_lat, end_lon)
        
        # Calculate average values for the route (6 features)
        features = [
            (start_weather['temperature'] + end_weather['temperature']) / 2,  # Temperature
            (start_weather['humidity'] + end_weather['humidity']) / 2,        # Humidity
            (start_weather['rainfall'] + end_weather['rainfall']) / 2,        # Rainfall
            (start_weather['pressure'] + end_weather['pressure']) / 2,        # Pressure
            (start_weather['wind_speed'] + end_weather['wind_speed']) / 2,    # Wind Speed
            (start_tide + end_tide) / 2                                       # Tide Level
        ]
        
        return features, start_weather, end_weather, start_tide, end_tide
        
    except Exception as e:
        raise Exception(f"Error generating features: {str(e)}")

def predict_weather_risk(start_location, end_location, model):
    """Predict flood risk using real weather data"""
    try:
        features, start_weather, end_weather, start_tide, end_tide = generate_features_from_locations(
            start_location, end_location
        )
        
        # Make prediction using the trained model
        prediction = model.predict([features])[0]
        
        # Get prediction probability (if available)
        try:
            prediction_proba = model.predict_proba([features])[0]
            confidence = max(prediction_proba) * 100
        except:
            confidence = None
        
        # Convert prediction to readable format
        if prediction == 1:
            result = "🔴 HIGH FLOOD RISK - Consider alternative route"
            risk_level = "High"
        else:
            result = "🟢 LOW FLOOD RISK - Route appears safe"
            risk_level = "Low"
            
        return result, risk_level, confidence, features, start_weather, end_weather, start_tide, end_tide
        
    except Exception as e:
        return f"Unable to analyze route: {str(e)}", None, None, None, None, None, None, None

# Map-related functions (from your existing code)
def calculate_route_risk(start_location, end_location, df, route_type="direct"):
    """Calculate overall risk for a route between two locations"""
    start_data = df[df['Location'] == start_location].iloc[0] if not df[df['Location'] == start_location].empty else None
    end_data = df[df['Location'] == end_location].iloc[0] if not df[df['Location'] == end_location].empty else None
    
    if start_data is None or end_data is None:
        return "Unknown", 0.5
    
    # Get risk scores
    start_risk = get_risk_score(start_data.get('FloodRiskLevel', 'None'), start_data.get('RoadRiskLevel', 'None'))
    end_risk = get_risk_score(end_data.get('FloodRiskLevel', 'None'), end_data.get('RoadRiskLevel', 'None'))
    
    # Add route-specific modifiers
    route_modifiers = {
        "highway": 0.8,
        "direct": 1.0,
        "alternate": 1.2
    }
    
    avg_risk = (start_risk + end_risk) / 2 * route_modifiers.get(route_type, 1.0)
    
    if avg_risk > 0.7:
        return "High", avg_risk
    elif avg_risk > 0.4:
        return "Moderate", avg_risk
    else:
        return "Low", avg_risk

def get_risk_score(flood_risk, road_risk):
    """Convert risk levels to numeric scores"""
    risk_scores = {'severe': 1.0, 'high': 0.8, 'moderate': 0.5, 'low': 0.2, 'none': 0.0}
    flood_score = risk_scores.get(str(flood_risk).lower(), 0.3)
    road_score = risk_scores.get(str(road_risk).lower(), 0.3)
    return (flood_score + road_score) / 2

def generate_route_points(start_coords, end_coords, route_type="direct"):
    """Generate intermediate points for a route"""
    random.seed(hash((start_coords, end_coords, route_type)) % 1000000)
    
    start_lat, start_lon = start_coords
    end_lat, end_lon = end_coords
    
    if route_type == "direct":
        mid_lat = (start_lat + end_lat) / 2 + random.uniform(-0.005, 0.005)
        mid_lon = (start_lon + end_lon) / 2 + random.uniform(-0.005, 0.005)
        return [start_coords, (mid_lat, mid_lon), end_coords]
    
    elif route_type == "highway":
        highway_lat = start_lat + (end_lat - start_lat) * 0.3
        highway_lon = start_lon + (end_lon - start_lon) * 0.7
        return [start_coords, (highway_lat, highway_lon), end_coords]
    
    elif route_type == "alternate":
        alt_lat1 = start_lat + (end_lat - start_lat) * 0.2 + random.uniform(-0.01, 0.01)
        alt_lon1 = start_lon + (end_lon - start_lon) * 0.3 + random.uniform(-0.01, 0.01)
        alt_lat2 = start_lat + (end_lat - start_lat) * 0.8 + random.uniform(-0.008, 0.008)
        alt_lon2 = start_lon + (end_lon - start_lon) * 0.9 + random.uniform(-0.012, 0.012)
        return [start_coords, (alt_lat1, alt_lon1), (alt_lat2, alt_lon2), end_coords]

def get_route_color(risk_level):
    """Get color for route based on risk level"""
    colors = {
        "Low": "#00ff00",      # Green
        "Moderate": "#ffa500", # Orange  
        "High": "#ff0000"      # Red
    }
    return colors.get(risk_level, "#0000ff")

# Main application
def main():
    # Title
    st.title("🌊 FloodSight - Complete Route Analysis")
    st.markdown("### AI-Powered Flood Risk Analysis with Interactive Route Mapping")
    
    # Load data
    model = load_model()
    df = load_flood_data()
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["🤖 AI Weather Prediction", "🗺️ Interactive Route Map"])
    
    with tab1:
        st.subheader("Real-Time Weather Based Flood Prediction")
        
        # Get location list from loaded df
        if df is not None and 'Location' in df.columns:
            location_list = sorted(df['Location'].unique().tolist())
        else:
            location_list = []
        
        # Weather prediction interface
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 Starting Location")
            if location_list:
                start_location_weather = st.selectbox(
                    "Select starting location",
                    ["Select location"] + location_list,
                    key="start_weather_dropdown"
                )
                # Convert "Select location" to empty string for processing
                start_location_weather = "" if start_location_weather == "Select location" else start_location_weather
            else:
                start_location_weather = st.text_input(
                    "Enter starting location (CSV not loaded)",
                    placeholder="e.g., Mumbai, Maharashtra",
                    key="start_weather_text"
                )

        with col2:
            st.subheader("🏁 Destination")
            if location_list:
                end_location_weather = st.selectbox(
                    "Select destination",
                    ["Select location"] + location_list,
                    key="end_weather_dropdown"
                )
                # Convert "Select location" to empty string for processing
                end_location_weather = "" if end_location_weather == "Select location" else end_location_weather
            else:
                end_location_weather = st.text_input(
                    "Enter destination (CSV not loaded)",
                    placeholder="e.g., Thane, Maharashtra",
                    key="end_weather_text"
                )


            # Weather analysis button
        if st.button("🔍 Analyze Weather Risk", type="primary", use_container_width=True):
            if start_location_weather and end_location_weather:
                if model is not None:
                    with st.spinner("Getting weather data and analyzing flood risk..."):
                        result, risk_level, confidence, features, start_weather, end_weather, start_tide, end_tide = predict_weather_risk(
                            start_location_weather, end_location_weather, model
                        )
                    
                    if risk_level:
                        st.success("✅ Weather Analysis Complete!")
                        
                        # Main result
                        if risk_level == "High":
                            st.error(result)
                        else:
                            st.success(result)
                        
                        # Display weather details and features (same as before)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Risk Level", risk_level)
                        with col2:
                            if confidence:
                                st.metric("Confidence", f"{confidence:.1f}%")
                            else:
                                st.metric("Model Score", f"{features[0]:.1f}")
                        with col3:
                            st.metric("Analysis", "Complete")
                        
                        # Weather details (same as your original code)
                        st.subheader("🌤️ Current Weather Conditions")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**📍 {start_location_weather}**")
                            if start_weather:
                                st.metric("🌡️ Temperature", f"{start_weather['temperature']:.1f}°C")
                                st.metric("💧 Humidity", f"{start_weather['humidity']:.0f}%")
                                st.metric("🌧️ Rainfall", f"{start_weather['rainfall']:.1f}mm")
                                st.metric("🌪️ Pressure", f"{start_weather['pressure']:.0f}hPa")
                                st.metric("💨 Wind Speed", f"{start_weather['wind_speed']:.1f}m/s")
                                st.metric("🌊 Tide Level", f"{start_tide:.2f}m")
                        
                        with col2:
                            st.markdown(f"**📍 {end_location_weather}**")
                            if end_weather:
                                st.metric("🌡️ Temperature", f"{end_weather['temperature']:.1f}°C")
                                st.metric("💧 Humidity", f"{end_weather['humidity']:.0f}%")
                                st.metric("🌧️ Rainfall", f"{end_weather['rainfall']:.1f}mm")
                                st.metric("🌪️ Pressure", f"{end_weather['pressure']:.0f}hPa")
                                st.metric("💨 Wind Speed", f"{end_weather['wind_speed']:.1f}m/s")
                                st.metric("🌊 Tide Level", f"{end_tide:.2f}m")
                    else:
                        st.error(result)
                else:
                    st.error("❌ Model could not be loaded.")
            else:
                st.warning("⚠️ Please enter both locations.")
    
    with tab2:
        st.subheader("Interactive Route Planning with Local Data")
        
        if df is not None:
            # Create two columns for the interface
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Route Planning")
                
                # Get unique locations for dropdown
                location_list = sorted(df['Location'].unique().tolist()) if 'Location' in df.columns else []
                
                # From and To selectors
                from_location = st.selectbox("From:", ["Select location"] + location_list, key="from_map")
                to_location = st.selectbox("To:", ["Select location"] + location_list, key="to_map")
                
                # Route options
                st.subheader("Route Options")
                show_direct = st.checkbox("Direct Route", value=True)
                show_highway = st.checkbox("Highway Route", value=True) 
                show_alternate = st.checkbox("Alternate Route", value=True)
                
                # Calculate routes button
                if st.button("Find Routes") and from_location != "Select location" and to_location != "Select location":
                    st.session_state.routes_calculated = True
                    st.session_state.from_loc = from_location
                    st.session_state.to_loc = to_location

            with col2:
                st.subheader("Route Map")
                
                # Initialize map centered on Mumbai/Thane
                map_center = [19.2183, 72.9781]
                m = folium.Map(location=map_center, zoom_start=11)
                
                
                # Add all location markers (only if coordinate columns exist)
                if df is not None:
                    # Try to find coordinate columns
                    lat_col = None
                    lon_col = None
                    
                    # Common variations of latitude/longitude column names
                    lat_variations = ['Latitude', 'latitude', 'lat', 'Lat', 'LAT', 'y', 'Y']
                    lon_variations = ['Longitude', 'longitude', 'lon', 'Lon', 'LON', 'lng', 'x', 'X']
                    
                    for col in df.columns:
                        if any(var.lower() in col.lower() for var in lat_variations):
                            lat_col = col
                        if any(var.lower() in col.lower() for var in lon_variations):
                            lon_col = col
                    
                    if lat_col and lon_col:
                        st.success(f"✅ Found coordinates: {lat_col}, {lon_col}")
                        
                        # Add location markers
                        for _, row in df.iterrows():
                            if pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
                                folium.CircleMarker(
                                    location=[row[lat_col], row[lon_col]],
                                    radius=4,
                                    color='lightblue',
                                    fill=True,
                                    fillColor='lightblue',
                                    fillOpacity=0.3,
                                    popup=f"{row.get('Location', 'Unknown')}",
                                    tooltip=row.get('Location', 'Unknown')
                                ).add_to(m)
                    else:
                        st.error("❌ No coordinate columns found in the data")
                        st.write("💡 Your CSV needs latitude and longitude columns for mapping")
                
                # Initialize routes_info as empty list
                routes_info = []
                
                # If routes are calculated, display them
                if hasattr(st.session_state, 'routes_calculated') and st.session_state.routes_calculated and df is not None:
                    from_loc = st.session_state.from_loc
                    to_loc = st.session_state.to_loc
                    
                    # Get coordinates for from/to locations (with error handling)
                    try:
                        from_data = df[df['Location'] == from_loc].iloc[0]
                        to_data = df[df['Location'] == to_loc].iloc[0]
                        
                        # Use the detected column names
                        if lat_col and lon_col:
                            from_coords = (from_data[lat_col], from_data[lon_col])
                            to_coords = (to_data[lat_col], to_data[lon_col])
                            
                            # Add prominent markers for start/end
                            folium.Marker(
                                from_coords,
                                popup=f"START: {from_loc}",
                                icon=folium.Icon(color='green', icon='play')
                            ).add_to(m)
                            
                            folium.Marker(
                                to_coords, 
                                popup=f"END: {to_loc}",
                                icon=folium.Icon(color='red', icon='stop')
                            ).add_to(m)
                            
                            # Generate and display routes
                            if show_direct:
                                risk_level, risk_score = calculate_route_risk(from_loc, to_loc, df, "direct")
                                route_points = generate_route_points(from_coords, to_coords, "direct")
                                color = get_route_color(risk_level)
                                
                                folium.PolyLine(
                                    route_points,
                                    color=color,
                                    weight=6,
                                    opacity=0.8,
                                    popup=f"Direct Route - {risk_level} Risk"
                                ).add_to(m)
                                routes_info.append(("Direct Route", risk_level, risk_score, "Shortest path"))
                            
                            if show_highway:
                                risk_level, risk_score = calculate_route_risk(from_loc, to_loc, df, "highway")
                                route_points = generate_route_points(from_coords, to_coords, "highway")
                                color = get_route_color(risk_level)
                                
                                folium.PolyLine(
                                    route_points,
                                    color=color,
                                    weight=6,
                                    opacity=0.8,
                                    popup=f"Highway Route - {risk_level} Risk",
                                    dashArray="10, 10"
                                ).add_to(m)
                                routes_info.append(("Highway Route", risk_level, risk_score, "Via major roads"))
                            
                            if show_alternate:
                                risk_level, risk_score = calculate_route_risk(from_loc, to_loc, df, "alternate")
                                route_points = generate_route_points(from_coords, to_coords, "alternate")
                                color = get_route_color(risk_level)
                                
                                folium.PolyLine(
                                    route_points,
                                    color=color,
                                    weight=6,
                                    opacity=0.8,
                                    popup=f"Alternate Route - {risk_level} Risk",
                                    dashArray="5, 15"
                                ).add_to(m)
                                routes_info.append(("Alternate Route", risk_level, risk_score, "Longer route"))
                            
                            # Center map on the route
                            center_lat = (from_coords[0] + to_coords[0]) / 2
                            center_lon = (from_coords[1] + to_coords[1]) / 2
                            m.location = [center_lat, center_lon]
                            m.zoom_start = 12
                        else:
                            st.error("❌ Cannot display routes - no coordinate columns found")
                            
                    except Exception as e:
                        st.error(f"❌ Error creating route: {e}")
                
                # Display the map
                st_folium(m, width=700, height=500)
            
            # Route information panel - ONLY show if routes_info has data
            if hasattr(st.session_state, 'routes_calculated') and st.session_state.routes_calculated and routes_info:
                st.subheader("📊 Route Analysis")
                
                route_data = []
                for route_name, risk_level, risk_score, description in routes_info:
                    route_data.append({
                        "Route": route_name,
                        "Risk Level": risk_level,
                        "Risk Score": f"{risk_score:.2f}",
                        "Description": description,
                        "Recommendation": "✅ Recommended" if risk_level == "Low" else "⚠️ Caution" if risk_level == "Moderate" else "❌ Avoid"
                    })
                
                route_df = pd.DataFrame(route_data)
                st.table(route_df)
                
                # Best route recommendation
                if routes_info:  # Check if routes_info is not empty
                    best_route = min(routes_info, key=lambda x: x[2])
                    st.success(f"🏆 **Recommended Route:** {best_route[0]} ({best_route[1]} Risk)")
                
                # Additional information
                st.subheader("🚨 Current Conditions")
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.write(f"**From:** {st.session_state.from_loc}")
                    try:
                        from_info = df[df['Location'] == st.session_state.from_loc].iloc[0]
                        st.write(f"- Flood Risk: {from_info.get('FloodRiskLevel', 'N/A')}")
                        st.write(f"- Road Risk: {from_info.get('RoadRiskLevel', 'N/A')}")
                        st.write(f"- Rainfall: {from_info.get('Rainfall', 'N/A')}mm")
                    except:
                        st.write("- Data not available")
                
                with col_info2:
                    st.write(f"**To:** {st.session_state.to_loc}")
                    try:
                        to_info = df[df['Location'] == st.session_state.to_loc].iloc[0]
                        st.write(f"- Flood Risk: {to_info.get('FloodRiskLevel', 'N/A')}")
                        st.write(f"- Road Risk: {to_info.get('RoadRiskLevel', 'N/A')}")
                        st.write(f"- Rainfall: {to_info.get('Rainfall', 'N/A')}mm")
                    except:
                        st.write("- Data not available")

            # Legend
            st.markdown("---")
            st.subheader("🗺️ Route Legend")
            col_legend1, col_legend2, col_legend3 = st.columns(3)
            with col_legend1:
                st.markdown("🟢 **Low Risk Route** (Recommended)")
            with col_legend2:
                st.markdown("🟡 **Moderate Risk Route** (Caution)")
            with col_legend3:
                st.markdown("🔴 **High Risk Route** (Avoid if possible)")
        else:
            st.error("❌ Could not load flood data CSV file.")

if __name__ == "__main__":
    main()
