import pandas as pd
from bs4 import BeautifulSoup
import re

# The HTML data from your table
html_data = """
<table>
  <caption><strong> NFL HEX Color Codes Table </strong></caption>
      <colgroup>
        <col span="1" class="nfl-team-name">
        <col span="1" class="color-1">
        <col span="1" class="color-2">
        <col span="1" class="color-3">
        <col span="1" class="color-4">
        <col span="1" class="color-5">
    </colgroup>
  <thead>
    <tr><th scope="col"> NFL Team Name </th>
    <th scope="col"> Color 1</th>
    <th scope="col"> Color 2</th>
    <th scope="col"> Color 3</th>
    <th scope="col"> Color 4</th>
    <th scope="col"> Color 5</th>
  </tr></thead>
<tbody>
  <tr><th scope="row"> Baltimore Ravens</th> <td> Purple #241773</td> <td> Black #000000</td> <td> Gold #9E7C0C </td> <td> Red #C60C30</td><td> </td></tr>
  <tr><th scope="row"> Cincinnati Bengals</th> <td> Orange #FB4F14</td> <td> Black #000000</td> <td> </td> <td> </td><td> </td></tr> 
  <tr><th scope="row"> Cleveland Browns</th> <td> Brown #311D00</td> <td> Orange #FF3C00</td> <td> </td> <td> </td><td> </td></tr>
  <tr><th scope="row"> Pittsburgh Steelers</th> <td> Steelers Gold #FFB612</td> <td> Black #101820</td> <td> Blue #003087</td> <td> Red #C60C30</td><td> Silver #A5ACAF </td></tr>
  <tr><th scope="row"> Buffalo Bills</th> <td> Blue #00338D </td> <td> Red #C60C30</td> <td> </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Miami Dolphins</th> <td> Aqua #008E97</td> <td> Orange #FC4C02</td> <td> Blue #005778</td> <td> </td><td> </td></tr>
<tr><th scope="row"> New England Patriots</th> <td> Nautical Blue #002244</td><td> Red #C60C30</td>  <td> New Century Silver #B0B7BC </td> <td> </td><td> </td></tr>
<tr><th scope="row"> New York Jets</th> <td> Gotham Green #125740</td><td> Stealth Black #000000</td>  <td> Spotlight White #FFFFFF </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Houston Texans</th> <td> Deep Steel Blue #03202F </td><td> Battle Red #A71930</td>  <td> </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Indianapolis Colts</th> <td> Speed Blue #002C5F </td> <td> Gray #A2AAAD </td> <td> </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Jacksonville Jaguars </th> <td> Black #101820</td><td> Gold #D7A22A </td>  <td> Dark Gold #9F792C </td> <td> Teal #006778</td><td> </td></tr>
<tr><th scope="row"> Tennessee Titans</th> <td> Titans Navy #0C2340</td><td> Titans Blue #4B92DB </td>  <td> Titans Red #C8102E </td> <td> Titans Silver #8A8D8F </td><td> </td></tr>
<tr><th scope="row"> Denver Broncos</th> <td> Broncos Orange #FB4F14</td> <td> Broncos Navy #002244</td> <td> </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Kansas City Chiefs</th> <td> Red #E31837</td><td> Gold #FFB81C </td>  <td> </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Las Vegas Raiders</th> <td> Raiders Black #000000</td><td> Raiders Silver #A5ACAF </td>  <td> </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Los Angeles Chargers </th> <td> Powder Blue #0080C6</td><td> Sunshine Gold #FFC20E </td>  <td> White #FFFFFF </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Chicago Bears</th> <td> Dark Navy #0B162A </td> <td> Orange #C83803</td> <td> </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Detroit Lions</th> <td> Honolulu Blue #0076B6</td><td> Silver #B0B7BC </td> <td> Black #000000</td> <td> White #FFFFFF </td><td> </td></tr>
<tr><th scope="row"> Green Bay Packers </th> <td> Dark Green #203731</td><td> Gold #FFB612</td>  <td> </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Minnesota Vikings </th> <td> Purple #4F2683</td> <td> Gold #FFC62F </td> <td> </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Dallas Cowboys </th> <td> Royal Blue #003594</td> <td> Blue #041E42</td> <td> Silver #869397</td> <td> Silver-Green #7F9695</td><td> White #FFFFFF </td></tr>
<tr><th scope="row"> New York Giants </th> <td> Dark Blue #0B2265</td> <td> Red #A71930</td> <td> Gray #A5ACAF </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Philadelphia Eagles </th> <td> Midnight Green #004C54</td> <td> Silver (Jersey) #A5ACAF </td> <td> Silver (Helmet) #ACC0C6</td> <td> Black #000000</td><td> Charcoal #565A5C </td></tr>
<tr><th scope="row"> Washington Commanders </th> <td> Burgundy #5A1414</td> <td> Gold #FFB612</td> <td> </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Atlanta Falcons</th> <td> Red #A71930</td><td> Black #000000</td>  <td> Silver #A5ACAF </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Carolina Panthers</th> <td> Carolina Blue #0085CA </td><td> Black #101820</td>  <td> Silver #BFC0BF </td> <td> </td><td> </td></tr>
<tr><th scope="row"> New Orleans Saints</th> <td> Old Gold #D3BC8D </td> <td> Black #101820</td> <td> </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Tampa Bay Buccaneers</th> <td> Red #D50A0A </td><td> Bay Orange #FF7900</td>  <td> Black #FF7900</td> <td> Grey #B1BABF </td><td> Pewter #34302B </td></tr>
<tr><th scope="row"> Arizona Cardinals</th> <td> Red #97233F </td><td> Black #000000</td>  <td> Yellow #FFB612</td> <td> </td><td> </td></tr>
<tr><th scope="row"> Los Angeles Rams</th> <td> Blue #003594</td> <td> Gold #FFA300</td> <td> Dark Gold #FF8200</td> <td> Yellow #FFD100</td><td> White #FFFFFF </td></tr>
<tr><th scope="row"> San Francisco 49ers</th> <td> 49ers Red #AA0000</td> <td> Gold #B3995D </td> <td> </td> <td> </td><td> </td></tr>
<tr><th scope="row"> Seattle Seahawks</th> <td> College Navy #002244</td> <td> Action Green #69BE28</td> <td> Wolf Gray #A5ACAF </td> <td> </td><td> </td></tr>
</tbody>
</table>
"""

# This dictionary now reflects the new abbreviations you provided.
team_name_to_abbr = {
    "Pittsburgh Steelers": "PIT", "Chicago Bears": "CHI", "Green Bay Packers": "GB", "Atlanta Falcons": "ATL",
    "Houston Texans": "HOU", "Minnesota Vikings": "MIN",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Jacksonville Jaguars": "JAX", "Las Vegas Raiders": "LV", "Detroit Lions": "DET", "Arizona Cardinals": "ARI", "Seattle Seahawks": "SEA", "Los Angeles Rams": "LA",
    "Philadelphia Eagles": "PHI", "Cleveland Browns": "CLE", "New York Jets": "NYJ", "Tampa Bay Buccaneers": "TB",
    "Kansas City Chiefs": "KC", "Cincinnati Bengals": "CIN", "Tennessee Titans": "TEN", "Dallas Cowboys": "DAL",
    "Indianapolis Colts": "IND", "New Orleans Saints": "NO",
    "Miami Dolphins": "MIA", "Denver Broncos": "DEN", "Los Angeles Chargers": "LAC", "Washington Commanders": "WAS",
    "New York Giants": "NYG", "New England Patriots": "NE", "Carolina Panthers": "CAR", "San Francisco 49ers": "SF"
}

def extract_hex_code(cell_text):
    """Finds and returns the HEX code from a string, or an empty string if not found."""
    match = re.search(r'#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})', cell_text)
    return match.group(0) if match else ''

# Use BeautifulSoup to parse the HTML
soup = BeautifulSoup(html_data, 'html.parser')

# Find all the rows in the table body
table_rows = soup.find('tbody').find_all('tr')

# A list to hold all the processed data for each team
processed_data = []

# Loop through each row in the table
for row in table_rows:
    # Get the team name from the row header (th)
    full_team_name = row.find('th').get_text(strip=True)
    
    # Get the abbreviation from our map, defaulting to the full name if not found
    team_abbr = team_name_to_abbr.get(full_team_name, full_team_name)
    
    # Get all the color data cells (td)
    color_cells = row.find_all('td')
    
    # Extract the hex code from each color cell
    hex_codes = [extract_hex_code(cell.get_text()) for cell in color_cells]
    
    # Create a dictionary for the current row's data
    row_data = {
        'Team': team_abbr,
        'Color 1': hex_codes[0],
        'Color 2': hex_codes[1],
        'Color 3': hex_codes[2],
        'Color 4': hex_codes[3],
        'Color 5': hex_codes[4],
    }
    processed_data.append(row_data)

# Create a pandas DataFrame from our list of dictionaries
df = pd.DataFrame(processed_data)

# Convert the DataFrame to a CSV string, without the pandas index
csv_output = df.to_csv(index=False)

# Print the final CSV output
print(csv_output)