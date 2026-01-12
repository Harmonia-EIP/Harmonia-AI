import json
import ast

# The 9 parameters for your JUCE plugin
TARGET_PARAMS = [
    'LFO 1 Rate',       # 1. Frequency
    'AmpEnv A Attack',  # 2. Attack
    'Filter A Cutoff',  # 3. Cutoff
    'AmpEnv A Decay',   # 4. Decay
    'Main Volume',      # 5. Volume
    'AmpEnv A Sustain', # 6. Sustain
    'Filter A Reso',    # 7. Resonance
    'AmpEnv A Release', # 8. Release
    'Osc A1 Waveform'   # 9. Waveform
]

def clean_content(content):
    """
    Attempts to fix the string format so it becomes a valid Python tuple.
    """
    content = content.strip()

    # remove trailing comma if present
    if content.endswith(','):
        content = content[:-1]

    # Fix unbalanced parentheses
    has_start = content.startswith('(')
    has_end = content.endswith(')')

    if has_end and not has_start:
        # Case: "Name', {...})" -> Add missing start "("
        content = '(' + content
    elif has_start and not has_end:
        # Case: "('Name', {...}" -> Add missing end ")"
        content = content + ')'
    elif not has_start and not has_end:
        # Case: "'Name', {...}" -> Add both
        content = f"({content})"

    return content

def convert_fxp_dump_to_json(input_file, output_file):
    dataset = []

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_content = f.read()

    # 1. Try to clean up the text format
    fixed_content = clean_content(raw_content)

    try:
        # 2. Parse the text into actual Python objects
        parsed_data = ast.literal_eval(fixed_content)

        # Ensure we have a list of entries to iterate over
        # If the file contains just one tuple, wrap it in a list
        if isinstance(parsed_data, tuple):
            raw_data = [parsed_data]
        elif isinstance(parsed_data, list):
            raw_data = parsed_data
        else:
            print(f"Unexpected data format: {type(parsed_data)}")
            return

    except Exception as e:
        print(f"Error parsing file: {e}")
        print("Tip: Check if your text file starts with '(' and ends with ')'.")
        return

    # 3. Extract the specific knobs we want
    for entry in raw_data:
        # Handle cases where the tuple might be (name, data) or just data
        if isinstance(entry, tuple) and len(entry) == 2:
            name, params_dict = entry
        elif isinstance(entry, dict):
            # Fallback if the file is just a list of dicts
            params_dict = entry
            name = params_dict.get('description', 'Unknown Preset')
        else:
            continue

        # Clean up description
        description = params_dict.get('description', name)
        description = str(description).replace('.fxp', '').replace('_', ' ')

        # Extract values
        extracted_values = []
        for key in TARGET_PARAMS:
            val = params_dict.get(key, 0.0)
            extracted_values.append(float(val))

        dataset.append({
            "description": description,
            "parameters": extracted_values
        })

    # 4. Save to JSON
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f"Success! Converted {len(dataset)} preset(s) to {output_file}")

if __name__ == "__main__":
    # Make sure this filename matches your text file
    convert_fxp_dump_to_json("../data/raw/my_raw_dump.txt", "presets.json")
