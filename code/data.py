SUMMARY = 'summary'
LOG = 'log'


def get_record_type(key):
    if key.startswith("#summary:#"):
        return SUMMARY
    else:
        return LOG


def load_data(file):
    data_log = {}
    data_summary = []
    current_key = None
    lines = []

    with open(file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                if current_key is not None:
                    if get_record_type(current_key) == SUMMARY:
                        data_summary.append(lines)
                    else:
                        data_log[current_key.replace("#", "")] = lines

                current_key = line
                lines = []
            else:
                lines.append(line)

        if current_key is not None:
            data_summary.append(lines)
    return data_log, data_summary
