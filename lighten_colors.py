import json

# Your original JavaScript object as a string
js_input_string = """
const TEAM_COLORS = {
    "BAL": { background: "#241773", text: "#FFFFFF" },
    "CIN": { background: "#FB4F14", text: "#000000" },
    "CLE": { background: "#311D00", text: "#FF3C00" },
    "PIT": { background: "#FFB612", text: "#101820" },
    "BUF": { background: "#00338D", text: "#C60C30" },
    "MIA": { background: "#008E97", text: "#FC4C02" },
    "NE": { background: "#002244", text: "#C60C30" },
    "NYJ": { background: "#125740", text: "#FFFFFF" },
    "HOU": { background: "#03202F", text: "#A71930" },
    "IND": { background: "#002C5F", text: "#A2AAAD" },
    "JAX": { background: "#101820", text: "#D7A22A" },
    "TEN": { background: "#0C2340", text: "#4B92DB" },
    "DEN": { background: "#FB4F14", text: "#002244" },
    "KC": { background: "#E31837", text: "#FFB81C" },
    "LV": { background: "#000000", text: "#A5ACAF" },
    "LAC": { background: "#0080C6", text: "#FFC20E" },
    "CHI": { background: "#0B162A", text: "#C83803" },
    "DET": { background: "#0076B6", text: "#B0B7BC" },
    "GB": { background: "#203731", text: "#FFB612" },
    "MIN": { background: "#4F2683", text: "#FFC62F" },
    "DAL": { background: "#003594", text: "#FFFFFF" },
    "NYG": { background: "#0B2265", text: "#A71930" },
    "PHI": { background: "#004C54", text: "#A5ACAF" },
    "WAS": { background: "#5A1414", text: "#FFB612" },
    "ATL": { background: "#A71930", text: "#000000" },
    "CAR": { background: "#0085CA", text: "#101820" },
    "NO": { background: "#D3BC8D", text: "#101820" },
    "TB": { background: "#D50A0A", text: "#FF7900" },
    "ARI": { background: "#97233F", text: "#FFFFFF" },
    "LA": { background: "#003594", text: "#FFA300" },
    "SF": { background: "#AA0000", text: "#B3995D" },
    "SEA": { background: "#002244", text: "#69BE28" }
};
"""

# The percentage to lighten the colors
LIGHTENING_PERCENTAGE = 30

def lighten_hex_color(hex_color, percent):
    """
    Lightens a hex color by a given percentage.
    """
    # Remove the '#' if it exists
    hex_color = hex_color.lstrip('#')

    # Convert the hex color to RGB values
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Calculate the new, lighter RGB values
    r = int(min(255, r + (255 - r) * (percent / 100)))
    g = int(min(255, g + (255 - g) * (percent / 100)))
    b = int(min(255, b + (255 - b) * (percent / 100)))

    # Convert the new RGB values back to hex
    return f'#{r:02x}{g:02x}{b:02x}'.upper()

def process_colors():
    """
    Parses the input string, lightens the colors, and generates the new JS object.
    """
    # Simple parsing to extract the content of the object
    # This is a bit brittle, but works for the provided format
    start_index = js_input_string.find('{')
    end_index = js_input_string.rfind('}') + 1
    
    # We need to replace the single quotes and keywords for JSON compatibility
    json_string = js_input_string[start_index:end_index].replace('\'', '\"').replace(' ', '').replace('\n', '').replace('background:', '\"background\":').replace('text:', '\"text\":')
    
    # Correctly format the keys in the object
    json_string = json_string.replace('\"', '\'')
    json_string = json_string.replace('\'background\'', '\"background\"').replace('\'text\'', '\"text\"')
    json_string = json_string.replace('\'#', '\"#')
    json_string = json_string.replace('}\'', '}\"').replace(',\'', ',\'').replace('\'', '\"')
    
    # A cleaner approach using regex would be more robust, but this works for the current format.
    json_string = js_input_string[start_index:end_index].strip().replace('\'', '"')
    json_string = json_string.replace('background:', '"background":').replace('text:', '"text":')
    json_string = json_string.replace('},', '},\n').replace(',\n', ',\n')
    
    # A simpler and more robust approach
    cleaned_js_string = js_input_string[js_input_string.find('{'):js_input_string.rfind('}')+1]
    cleaned_js_string = cleaned_js_string.replace('\'', '"').replace(':', ': ').replace(' ', '').replace('\n', '')
    
    # This is a robust way to handle this
    import re
    cleaned_js_string = re.sub(r'(\w+):', r'"\1":', js_input_string)
    cleaned_js_string = re.sub(r'\'', '"', cleaned_js_string)
    cleaned_js_string = cleaned_js_string[cleaned_js_string.find('{'):cleaned_js_string.rfind('}')+1]
    
    # Use ast.literal_eval for a safer way to parse the string
    import ast
    # A simple manual parsing approach:
    lines = [line.strip() for line in js_input_string.split('\n') if ':' in line]
    data = {}
    for line in lines:
        try:
            team_code, rest = line.split(':', 1)
            team_code = team_code.strip().replace('"', '').replace("'", "")
            rest = rest.replace('\'', '"').replace(' ', '').replace('//Changedtexttowhiteforreadability', '')
            rest = rest.replace('{', '').replace('}', '').replace('\'', '"')
            
            parts = rest.split(',')
            background_color = parts[0].split(':')[1].strip().replace('"', '')
            text_color = parts[1].split(':')[1].strip().replace('"', '')
            data[team_code] = {'background': background_color, 'text': text_color}
        except:
            continue
    
    # The dictionary to hold the new colors
    lighter_colors = {}
    
    # Process each color
    for team, colors in data.items():
        lighter_bg = lighten_hex_color(colors['background'], LIGHTENING_PERCENTAGE)
        lighter_colors[team] = {
            'background': lighter_bg,
            'text': colors['text']  # Keeping the text color the same
        }
        
    # Generate the new JavaScript object string
    js_output_string = 'const LIGHTER_TEAM_COLORS = {\n'
    for team, colors in lighter_colors.items():
        js_output_string += f'    "{team}": {{ background: "{colors["background"]}", text: "{colors["text"]}" }},\n'
    js_output_string = js_output_string.rstrip(',\n') + '\n};'
    
    # Write the new object to a file
    with open('lighter_team_colors.js', 'w') as f:
        f.write(js_output_string)
        
    print("New JavaScript file 'lighter_team_colors.js' has been generated!")

if __name__ == "__main__":
    process_colors()