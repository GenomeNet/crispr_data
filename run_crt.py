import subprocess
import argparse
import os
import re
from collections import Counter
import tempfile

def run_crt(input_file, output_file):
    cmd = ['java', '-cp', 'CRT1.2-CLI.jar', 'crt', input_file, output_file]
    try:
        subprocess.run(cmd, check=True)
        print(f"CRT run successfully on {input_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running CRT on {input_file}: {e}")
        raise

def parse_crt_output(crt_output):
    with open(crt_output, 'r') as f:
        lines = f.readlines()

    organism = None
    crisprs = []
    current_crispr = None
    repeats_collected = []

    for line in lines:
        line = line.strip()

        if line.startswith("ORGANISM:"):
            organism = line.split(":")[1].strip()
            print(f"Organism ID found: {organism}")
            continue

        crispr_range_match = re.match(r'^CRISPR\s+\d+\s+Range:\s+(\d+)\s*-\s*(\d+)', line)
        if crispr_range_match:
            if current_crispr:
                if repeats_collected:
                    repeat_seq = Counter(repeats_collected).most_common(1)[0][0]
                    current_crispr['repeat_seq'] = repeat_seq
                    crisprs.append(current_crispr)
                    print(f"Finalized CRISPR {current_crispr['id']} with repeat sequence: {repeat_seq}")
                repeats_collected = []

            current_crispr = {
                'id': len(crisprs) + 1,
                'start': int(crispr_range_match.group(1)),
                'end': int(crispr_range_match.group(2)),
                'num_repeats': 0,
                'repeat_seq': ''
            }
            print(f"Detected CRISPR {current_crispr['id']} with range: {current_crispr['start']} - {current_crispr['end']}")
            continue

        repeats_match = re.match(r'^Repeats:\s+(\d+)', line)
        if repeats_match and current_crispr:
            current_crispr['num_repeats'] = int(repeats_match.group(1))
            print(f"CRISPR {current_crispr['id']} has {current_crispr['num_repeats']} repeats")
            continue

        if current_crispr and line and not line.startswith("POSITION"):
            parts = re.split(r'\s+', line)
            if len(parts) >= 2:
                repeat = parts[1]
                repeats_collected.append(repeat)
                print(f"CRISPR {current_crispr['id']} - Repeat: {repeat}")
            continue

    if current_crispr and repeats_collected:
        repeat_seq = Counter(repeats_collected).most_common(1)[0][0]
        current_crispr['repeat_seq'] = repeat_seq
        crisprs.append(current_crispr)
        print(f"Finalized CRISPR {current_crispr['id']} with repeat sequence: {repeat_seq}")

    return organism, crisprs
def format_gff(organism, crisprs):
    gff_lines = []
    for crispr in crisprs:
        gff_line = (f"{organism}\tcrt:1.2\trepeat_region\t{crispr['start']}\t{crispr['end']}\t.\t.\t.\t"
                    f"note=CRISPR with {crispr['num_repeats']} repeat units;rpt_family=CRISPR;"
                    f"rpt_type=direct;rpt_unit_seq={crispr['repeat_seq']}")
        gff_lines.append(gff_line)
    return '\n'.join(gff_lines)

def integrate_gff(original_gff, new_gff_content, output_integrated_gff):
    with open(original_gff, 'r') as orig, open(output_integrated_gff, 'w') as out:
        # Copy original GFF content
        for line in orig:
            if line.startswith('##FASTA'):
                break
            out.write(line)
        
        # Add new CRT predictions
        new_entries_added = False
        for line in new_gff_content.split('\n'):
            if line:
                if not new_entries_added:
                    new_entries_added = True
                out.write(line + '\n')
        
        # Copy FASTA section if present
        orig.seek(0)
        fasta_section = False
        for line in orig:
            if line.startswith('##FASTA'):
                fasta_section = True
            if fasta_section:
                out.write(line)

def main():
    parser = argparse.ArgumentParser(description="Run CRT and convert output to GFF format")
    parser.add_argument("--input_folder", required=True, help="Input folder containing FASTA files")
    parser.add_argument("--output_folder", required=True, help="Output folder for CRT GFF files")
    parser.add_argument("--update_gff", required=True, help="Folder containing original GFF files to update")
    parser.add_argument("--output_gff", required=True, help="Output folder for integrated GFF files")

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.output_gff, exist_ok=True)

    for filename in os.listdir(args.input_folder):
        if filename.endswith('.fasta'):
            input_file = os.path.join(args.input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            crt_gff = os.path.join(args.output_folder, f"{base_name}.gff")
            original_gff = os.path.join(args.update_gff, f"{base_name}.gff")
            output_integrated_gff = os.path.join(args.output_gff, f"{base_name}.gff")

            print(f"Processing {filename}...")

            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_crt_output:
                run_crt(input_file, temp_crt_output.name)
                
                organism, crisprs = parse_crt_output(temp_crt_output.name)
                gff_content = format_gff(organism, crisprs)

            os.unlink(temp_crt_output.name)  # Delete the temporary file

            if crisprs:
                # Write CRT predictions to crt_gff
                with open(crt_gff, 'w') as f:
                    f.write(gff_content)
                print(f"CRT predictions written to: {crt_gff}")

                # Integrate with original GFF file
                if os.path.exists(original_gff):
                    integrate_gff(original_gff, gff_content, output_integrated_gff)
                    print(f"Integrated GFF file created: {output_integrated_gff}")
                else:
                    print(f"Original GFF file not found: {original_gff}")
                    # If original GFF doesn't exist, just copy the CRT predictions
                    with open(output_integrated_gff, 'w') as f:
                        f.write(gff_content)
                    print(f"New GFF file created: {output_integrated_gff}")

                print("Extracted CRISPR information:")
                for i, crispr in enumerate(crisprs, 1):
                    print(f"CRISPR {i}:")
                    print(f"  Range: {crispr['start']} - {crispr['end']}")
                    print(f"  Number of repeats: {crispr['num_repeats']}")
                    print(f"  Repeat sequence: {crispr['repeat_seq']}")
            else:
                print("No CRISPRs found.")
            
            print()

if __name__ == "__main__":
    main()