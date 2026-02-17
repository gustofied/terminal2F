from pathlib import Path
import json
import requests
import time

json_string = """
{
  "students": [
    {
      "id": 1,
      "name": "Timmy",
      "age": 21,
      "full-time": true
    },
    {
      "id": 1,
      "name": Joe",
      "age": 33,
      "full-time": false
    }
  ]
}
"""

# data = json.loads(json_string)
# print(data["students"])

BASE_URL = "https://formulae.brew.sh/api"

response = requests.get(f"{BASE_URL}/formula.json")
packages = response.json()

all_package_data = []

for package in packages[:10]:
    package_name = package["name"]
    package_desc = package["desc"]

    response_pkg = requests.get(f"{BASE_URL}/formula/{package_name}.json")
    package_data = response_pkg.json()

    with open("packagey.json", "w") as f:
        f.write(response_pkg.text)

    install_on_request = package_data["analytics"]["install_on_request"]
    installs_30d = install_on_request["30d"]
    installs_90d = install_on_request["90d"]
    installs_365d = install_on_request["365d"]

    data = {
        'name': package_name,
        'desc': package_desc,
        'analytics': {
            '30d': installs_30d,
            '60d': installs_90d,
            '365d': installs_365d,
        }
    }

    all_package_data.append(data)
    
    time.sleep(1)
    
    print(f'Got {package_name}')


dir = Path(__file__).resolve().parent
print(dir)

with open(f"{dir}/package_data.json", "w") as f:
    json.dump(all_package_data, f, indent=2)









