# main.py

import os
import openai
import overpy
import networkx as nx
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from scipy.spatial import cKDTree
import numpy as np
from dotenv import load_dotenv
import re
import logging
import time
import requests
from requests.exceptions import HTTPError, Timeout, ConnectionError
import hashlib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # To handle Cross-Origin requests

# Initialize Flask app with the correct static folder
# Ensure that 'frontend/public' is the correct path to your index.html
app = Flask(__name__, static_folder='frontend/public')
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Configure Logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs during development
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load Environment Variables
def load_environment():
    """Load environment variables from a .env file."""
    load_dotenv()
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    if not openai.api_key:
        logger.error("OpenAI API key not found in environment variables.")
        raise ValueError("OpenAI API key not found in environment variables.")

# Geocode Location
def geocode_location(location_name):
    """Geocode the location name to latitude and longitude."""
    geolocator = Nominatim(user_agent="cycling_route_planner_api")
    try:
        location = geolocator.geocode(location_name)
        if location:
            logger.info(f"Geocoded '{location_name}': ({location.latitude}, {location.longitude})")
            return (location.latitude, location.longitude)
        else:
            logger.error(f"Could not geocode location: {location_name}")
            return None
    except Exception as e:
        logger.error(f"Error during geocoding '{location_name}': {e}")
        return None

# Calculate Bounding Box
def calculate_bounding_box(point1, point2, padding=0.01):
    """Calculate a bounding box that includes both points with optional padding."""
    min_lat = min(point1[0], point2[0]) - padding
    max_lat = max(point1[0], point2[0]) + padding
    min_lon = min(point1[1], point2[1]) - padding
    max_lon = max(point1[1], point2[1]) + padding
    bbox = (min_lat, min_lon, max_lat, max_lon)
    logger.info(f"Calculated bounding box: {bbox}")
    return bbox

# OverpassQuery Class with Caching and Retries
class OverpassQuery:
    def __init__(self, cache_dir='overpass_cache', retries=3, backoff_factor=2, endpoint='https://overpass-api.de/api/interpreter'):
        self.api = overpy.Overpass(url=endpoint)
        self.cache_dir = cache_dir
        self.retries = retries
        self.backoff_factor = backoff_factor
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_filename(self, query):
        """Generate a cache filename based on the query hash."""
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{query_hash}.xml")  # Using XML format

    def query(self, query):
        """Query Overpass API with caching and retries."""
        cache_file = self.get_cache_filename(query)
        if os.path.exists(cache_file):
            logger.info("Loading data from cache.")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    xml_data = f.read()
                # Directly parse the XML data using Overpass API
                return self.api.parse_xml(xml_data)
            except Exception as e:
                logger.error(f"Error reading cache file: {e}")
                # If cache is corrupted, remove it and proceed to fetch again
                os.remove(cache_file)

        attempt = 0
        while attempt < self.retries:
            try:
                logger.info("Executing Overpass API query...")
                logger.debug(f"Query: {query.strip()}")
                result = self.api.query(query)
                # Save to cache
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(result.to_xml())
                logger.info("Query executed successfully and cached.")
                return result
            except (HTTPError, Timeout, ConnectionError) as e:
                wait = self.backoff_factor ** attempt
                logger.warning(f"Overpass API error: {e}. Retrying in {wait} seconds...")
                time.sleep(wait)
                attempt += 1
            except overpy.exception.OverpassTooManyRequests as e:
                wait = self.backoff_factor ** attempt
                logger.warning(f"Overpass API rate limit reached: {e}. Retrying in {wait} seconds...")
                time.sleep(wait)
                attempt += 1
            except overpy.exception.OverpassGatewayTimeout as e:
                wait = self.backoff_factor ** attempt
                logger.warning(f"Overpass API gateway timeout: {e}. Retrying in {wait} seconds...")
                time.sleep(wait)
                attempt += 1
            except Exception as e:
                logger.error(f"Unexpected error during Overpass API query: {e}")
                break
        logger.error("Failed to retrieve data from Overpass API after multiple attempts.")
        return None

# Query Overpass API Using overpy with Retries and Caching
def query_overpass_api(bbox, overpass_query):
    """Retrieve cycling-friendly ways and nodes within the bounding box from Overpass API."""
    min_lat, min_lon, max_lat, max_lon = bbox
    query = f"""
    [out:xml][timeout:60];
    (
      way["highway"="cycleway"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["highway"="path"]["bicycle"="yes"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["highway"="residential"]["bicycle"="yes"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["route"="bicycle"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    (._;>;);
    out body;
    """
    result = overpass_query.query(query)
    if result is None:
        logger.error("Overpass API query failed.")
        return []
    elements = []
    # Extract nodes
    for node in result.nodes:
        elements.append({
            'type': 'node',
            'id': node.id,
            'lat': node.lat,
            'lon': node.lon,
            'tags': node.tags
        })
    # Extract ways
    for way in result.ways:
        elements.append({
            'type': 'way',
            'id': way.id,
            'nodes': [node.id for node in way.nodes],
            'tags': way.tags
        })
    logger.info(f"Retrieved {len(elements)} elements from Overpass API.")
    return elements

# Build NetworkX Graph from OSM Data
def build_graph(elements):
    """Build a NetworkX graph from the retrieved OSM ways and nodes."""
    graph = nx.Graph()
    nodes = {}

    # Collect nodes with their coordinates
    for elem in elements:
        if elem['type'] == 'node':
            nodes[elem['id']] = (elem['lat'], elem['lon'])

    logger.info(f"Collected {len(nodes)} nodes.")

    # Build graph from ways
    for elem in elements:
        if elem['type'] == 'way':
            way_id = elem['id']
            node_ids = elem['nodes']
            for i in range(len(node_ids) - 1):
                node1_id = node_ids[i]
                node2_id = node_ids[i + 1]
                node1_coords = nodes.get(node1_id)
                node2_coords = nodes.get(node2_id)
                if node1_coords and node2_coords:
                    distance = geodesic(node1_coords, node2_coords).meters
                    graph.add_edge(node1_id, node2_id, weight=distance, osmid=way_id)
                    # Add node attributes
                    graph.nodes[node1_id]['lat'] = node1_coords[0]
                    graph.nodes[node1_id]['lon'] = node1_coords[1]
                    graph.nodes[node2_id]['lat'] = node2_coords[0]
                    graph.nodes[node2_id]['lon'] = node2_coords[1]
    logger.info(f"Processed {len([elem for elem in elements if elem['type'] == 'way'])} ways.")
    logger.info(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph

# Build KDTree for Nearest Neighbor Search
def build_kdtree(graph):
    """Build a KDTree for efficient nearest node search."""
    node_ids = list(graph.nodes())
    node_coords = np.array([
        (graph.nodes[node]['lat'], graph.nodes[node]['lon']) for node in node_ids
    ])
    kdtree = cKDTree(node_coords)
    logger.info("KDTree built for nearest neighbor search.")
    return kdtree, node_ids

# Find Nearest Node in Graph
def find_nearest_node(kdtree, node_ids, point):
    """Find the nearest node in the graph to the given point using KDTree."""
    distance, index = kdtree.query(point)
    nearest_node_id = node_ids[index]
    logger.info(f"Nearest node to point {point}: {nearest_node_id} at distance {distance:.2f} meters.")
    return nearest_node_id

# Find Routes Using NetworkX
def find_routes(graph, start_node, end_node, num_routes=5):
    """Find possible routes between start and end nodes using shortest path algorithm."""
    try:
        routes = list(nx.shortest_simple_paths(graph, source=start_node, target=end_node, weight='weight'))
        selected_routes = routes[:num_routes]
        logger.info(f"Found {len(selected_routes)} routes.")
        return selected_routes
    except nx.NetworkXNoPath:
        logger.error("No path found between the specified points.")
        return []
    except nx.NodeNotFound as e:
        logger.error(f"Node not found: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error while finding routes: {e}")
        return []

# Calculate Bike Path Percentage for a Route
def calculate_bike_path_percentage(graph, overpass_api, route):
    """Calculate the percentage of the route that is on dedicated bike paths."""
    total_length = 0
    bike_path_length = 0
    for i in range(len(route) - 1):
        node1 = route[i]
        node2 = route[i + 1]
        edge_data = graph.get_edge_data(node1, node2)
        if not edge_data:
            continue
        length = edge_data['weight']
        total_length += length

        way_id = edge_data.get('osmid')
        is_bike_path = False
        if way_id:
            try:
                # Fetch way details
                way = overpass_api.query(f"way({way_id}); out tags;")
                if way and way.ways:
                    highway = way.ways[0].tags.get("highway", "")
                    bicycle = way.ways[0].tags.get("bicycle", "")
                    if highway == 'cycleway' or bicycle in ['yes', 'designated']:
                        is_bike_path = True
            except Exception as e:
                logger.warning(f"Error retrieving way {way_id}: {e}")
        if is_bike_path:
            bike_path_length += length

    if total_length > 0:
        percentage = (bike_path_length / total_length) * 100
        logger.debug(f"Route bike path percentage: {percentage:.2f}%")
        return percentage
    else:
        return 0

# Fetch Elevation Data from Open-Elevation
def get_elevation(lat, lon, session=None):
    """Fetch elevation data for given latitude and longitude from Open-Elevation."""
    if session is None:
        session = requests.Session()
    try:
        response = session.get(
            'https://api.open-elevation.com/api/v1/lookup',
            params={'locations': f'{lat},{lon}'},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            elevation = data['results'][0]['elevation']
            return elevation
        else:
            logger.warning(f"Failed to get elevation for ({lat}, {lon}): Status {response.status_code}")
            return 0
    except Exception as e:
        logger.warning(f"Error fetching elevation for ({lat}, {lon}): {e}")
        return 0

# Calculate Elevation Gain for a Route
def calculate_elevation_gain(graph, route, session=None):
    """Calculate the total elevation gain for a route."""
    elevation_gain = 0
    previous_elevation = None
    for node_id in route:
        lat = graph.nodes[node_id]['lat']
        lon = graph.nodes[node_id]['lon']
        elevation = get_elevation(lat, lon, session=session)
        if previous_elevation is not None:
            gain = elevation - previous_elevation
            if gain > 0:
                elevation_gain += gain
        previous_elevation = elevation
    logger.debug(f"Total elevation gain: {elevation_gain} meters.")
    return elevation_gain

# Analyze Routes with OpenAI API
def analyze_routes_with_openai(routes_data):
    """Use OpenAI API to determine the safest route based on route attributes."""
    system_message = """
You are an expert in cycling safety and route planning.
Given a list of cycling routes with their attributes, analyze the routes
and select the safest route. Consider factors like length, elevation gain,
and bike path percentage. Provide the route number of the safest route
and a brief explanation of your reasoning.
    """

    user_message = "Here are the routes:\n\n"
    for data in routes_data:
        user_message += (
            f"Route {data['route_index']}:\n"
            f"- Length: {data['length']:.2f} meters\n"
            f"- Elevation Gain: {data['elevation_gain']:.2f} meters\n"
            f"- Bike Path Percentage: {data['bike_path_percentage']:.2f}%\n\n"
        )
    user_message += "Please select the safest route number and explain your reasoning."

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    try:
        logger.info("Sending analysis request to OpenAI API...")
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            max_tokens=150,
            temperature=0,
            n=1,
        )
        response_text = response.choices[0].message['content'].strip()
        logger.info("\nOpenAI Assistant's Response:")
        logger.info(response_text)

        # Simple parsing to extract the route number
        match = re.search(r'route\s+(\d+)', response_text, re.IGNORECASE)
        if match:
            selected_route_index = int(match.group(1))
            selected_route = next((route for route in routes_data if route['route_index'] == selected_route_index), None)
            return selected_route
        else:
            logger.error("Could not determine the safest route from the response.")
            return None
    except Exception as e:
        logger.error(f"An error occurred while calling the OpenAI API: {e}")
        return None

# Serve index.html at the root path
@app.route('/')
def serve_index():
    print("Serving index.html from:", app.static_folder)
    return send_from_directory(app.static_folder, 'index.html')

# API Endpoint
@app.route('/api/plan-route', methods=['POST'])
def plan_route():
    """
    Endpoint to plan a cycling route.
    Expects JSON payload with 'start_location' and 'end_location'.
    Returns JSON with possible routes and the safest route.
    """
    data = request.get_json()
    if not data:
        logger.error("Invalid JSON payload.")
        return jsonify({"error": "Invalid JSON payload."}), 400

    start_location = data.get('start_location')
    end_location = data.get('end_location')

    if not start_location or not end_location:
        logger.error("Both 'start_location' and 'end_location' are required.")
        return jsonify({"error": "Both 'start_location' and 'end_location' are required."}), 400

    # Geocode locations
    start_point = geocode_location(start_location)
    end_point = geocode_location(end_location)

    if not start_point or not end_point:
        logger.error("Failed to geocode one or both locations.")
        return jsonify({"error": "Failed to geocode one or both locations."}), 400

    # Calculate bounding box
    bbox = calculate_bounding_box(start_point, end_point)

    # Initialize OverpassQuery with caching and retries
    overpass_query = OverpassQuery(cache_dir='overpass_cache')  # Ensure cache directory is correctly set

    # Query Overpass API
    elements = query_overpass_api(bbox, overpass_query)
    if not elements:
        logger.error("No cycling-friendly ways found in the specified area.")
        return jsonify({"error": "No cycling-friendly ways found in the specified area."}), 404

    # Build graph
    graph = build_graph(elements)
    if graph.number_of_nodes() == 0:
        logger.error("Graph has no nodes. Possibly insufficient data.")
        return jsonify({"error": "Graph has no nodes. Possibly insufficient data."}), 500

    # Build KDTree
    kdtree, node_ids = build_kdtree(graph)

    # Find nearest nodes
    start_node = find_nearest_node(kdtree, node_ids, start_point)
    end_node = find_nearest_node(kdtree, node_ids, end_point)
    logger.info(f"Nearest node to start point: {start_node}")
    logger.info(f"Nearest node to end point: {end_node}")

    # Find routes
    routes = find_routes(graph, start_node, end_node)
    if not routes:
        logger.error("No routes found between the specified locations.")
        return jsonify({"error": "No routes found between the specified locations."}), 404

    # Analyze routes
    routes_data = []
    session = requests.Session()  # Reuse session for elevation requests
    for idx, route in enumerate(routes, start=1):
        length = sum(
            geodesic(
                (graph.nodes[route[i]]['lat'], graph.nodes[route[i]]['lon']),
                (graph.nodes[route[i + 1]]['lat'], graph.nodes[route[i + 1]]['lon'])
            ).meters for i in range(len(route) - 1)
        )
        elevation_gain = calculate_elevation_gain(graph, route, session=session)
        bike_path_percentage = calculate_bike_path_percentage(graph, overpass_query.api, route)
        routes_data.append({
            'route_index': idx,
            'route': route,
            'length': length,
            'elevation_gain': elevation_gain,
            'bike_path_percentage': bike_path_percentage,
        })

    # Analyze routes with OpenAI
    safest_route = analyze_routes_with_openai(routes_data)

    response = {
        'routes': routes_data,
        'safest_route': safest_route['route_index'] if safest_route else None
    }

    logger.info("Returning response with routes and safest_route.")
    return jsonify(response), 200

# Health Check Endpoint (Optional)
@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "API is running."}), 200

if __name__ == "__main__":
    try:
        load_environment()
    except Exception as e:
        logger.critical(f"Failed to load environment variables: {e}")
        exit(1)

    # Retrieve the port from environment variables (Replit sets this automatically)
    port = int(os.environ.get('PORT', 5000))

    # Run the Flask app on all available IPs and the specified port
    app.run(host='0.0.0.0', port=port, debug=True)
