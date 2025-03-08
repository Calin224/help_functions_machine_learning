### Import help_functions into the current working project:

```python
import requests
from pathlib import Path

if Path("helper_functions.py").is_file():
  print("The file already exists")
else:
  print("Downloading file...")
  request = requests.get("https://raw.githubusercontent.com/Calin224/help_functions_machine_learning/refs/heads/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)
```
