import requests

def make_post_request(url, form_data):
    try:
        response = requests.post(url, data=form_data)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            print("Request was successful!")
            print("Response data:", response.text)
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response data:", response.text)
            
    except requests.RequestException as e:
        print(f"Error making the request: {e}")

# Replace 'your_api_endpoint' with the actual API endpoint URL
api_url = 'http://192.168.251.119:8888/query'

# Replace 'your_key' and 'your_value' with the actual form data parameters
form_data = {
    'user_prompt': 'what is the rate of nissin noodles?'
}

make_post_request(api_url, form_data)
