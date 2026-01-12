import json
import ast

# 9 params of the front JUCE pluging
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
    content = content.strip()

    if content.endswith(','):
        content = content[:-1]

    has_start = content.startswith('(')
    has_end = content.endswith(')')

    if has_end and not has_start:
        content = '(' + content
    elif has_start and not has_end:
        content = content + ')'
    elif not has_start and not has_end:
        content = f"({content})"

    return content

def convert_fxp_dump_to_json(input_file, output_file):
    dataset = []

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_content = f.read()

    fixed_content = clean_content(raw_content)

    try:
        parsed_data = ast.literal_eval(fixed_content)

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

    for entry in raw_data:
        if isinstance(entry, tuple) and len(entry) == 2:
            name, params_dict = entry
        elif isinstance(entry, dict):
            params_dict = entry
            name = params_dict.get('description', 'Unknown Preset')
        else:
            continue

        description = params_dict.get('description', name)
        description = str(description).replace('.fxp', '').replace('_', ' ')

        extracted_values = []
        for key in TARGET_PARAMS:
            val = params_dict.get(key, 0.0)
            extracted_values.append(float(val))

        dataset.append({
            "description": description,
            "parameters": extracted_values
        })

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f"Success! Converted {len(dataset)} preset(s) to {output_file}")

if __name__ == "__main__":
    convert_fxp_dump_to_json("../data/raw/my_raw_dump.txt", "presets.json")
