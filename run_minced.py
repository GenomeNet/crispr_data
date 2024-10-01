import os
import subprocess
import argparse
import shutil

def run_minced(input_folder, output_folder, update_gff_folder, output_gff_folder):
    # Create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_gff_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.fasta'):
            input_file = os.path.join(input_folder, filename)
            output_base = os.path.splitext(filename)[0]
            output_gff = os.path.join(output_folder, f"{output_base}.gff")
            output_txt = os.path.join(output_folder, f"{output_base}.txt")

            # Run minced command
            cmd = ['minced/minced', '-gff', input_file, output_txt, output_gff]
            subprocess.run(cmd, check=True)

            # Count lines in the generated GFF file
            with open(output_gff, 'r') as f:
                line_count = sum(1 for _ in f)

            print(f"Processed {filename}")
            print(f"Generated GFF file: {output_gff}")
            print(f"Number of lines in GFF file: {line_count}")

            # If line count > 1, integrate with original GFF file
            if line_count > 1:
                original_gff = os.path.join(update_gff_folder, f"{output_base}.gff")
                output_integrated_gff = os.path.join(output_gff_folder, f"{output_base}.gff")

                if os.path.exists(original_gff):
                    with open(original_gff, 'r') as orig, open(output_gff, 'r') as new, open(output_integrated_gff, 'w') as out:
                        # Copy original GFF content and collect existing entries
                        existing_entries = set()
                        for line in orig:
                            if line.startswith('##FASTA'):
                                break
                            out.write(line)
                            if not line.startswith('#'):
                                parts = line.split('\t')
                                if len(parts) >= 5:
                                    key = (parts[0], parts[3], parts[4], parts[6])  # seqid, start, end, strand
                                    existing_entries.add(key)
                        
                        # Add minced predictions if not already present
                        new_entries_added = False
                        for line in new:
                            if not line.startswith('#'):
                                parts = line.split('\t')
                                if len(parts) >= 5:
                                    key = (parts[0], parts[3], parts[4], parts[6])
                                    if key not in existing_entries:
                                        if not new_entries_added:
                                            out.write("##CRISPR predictions from minced\n")
                                            new_entries_added = True
                                        out.write(line)
                                        existing_entries.add(key)
                        
                        # Copy FASTA section if present
                        orig.seek(0)
                        fasta_section = False
                        for line in orig:
                            if line.startswith('##FASTA'):
                                fasta_section = True
                            if fasta_section:
                                out.write(line)

                    print(f"Integrated GFF file created: {output_integrated_gff}")
                else:
                    print(f"Original GFF file not found: {original_gff}")
            else:
                print("No CRISPR predictions to integrate.")
            
            print()

def main():
    parser = argparse.ArgumentParser(description="Run minced on FASTA files, generate GFF files, and integrate with original GFF files")
    parser.add_argument("--input_folder", default="extracted_fasta", help="Input folder containing FASTA files")
    parser.add_argument("--output_folder", default="minced_gff", help="Output folder for minced GFF files")
    parser.add_argument("--update_gff", required=True, help="Folder containing original GFF files to update")
    parser.add_argument("--output_gff", required=True, help="Output folder for integrated GFF files")

    args = parser.parse_args()

    run_minced(args.input_folder, args.output_folder, args.update_gff, args.output_gff)

if __name__ == "__main__":
    main()