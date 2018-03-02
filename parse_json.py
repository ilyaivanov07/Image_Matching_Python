import json
from pprint import pprint

data = json.load(open('slack_json.json'))

# f = open("emails.txt","w+")

counter = 0
for member in data["members"]:
    if "email" in member["profile"]:
        counter += 1
        # f.write(member["profile"]["email"] + "\n")
        # print(member["profile"]["email"])

print(counter)

# f.close()

# pprint(data)

