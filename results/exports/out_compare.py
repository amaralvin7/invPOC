"""Compare if two text files are identical."""
import filecmp

f1 = 'test.txt'  # reference file
f2 = 'out.txt'  # new output file to be compared to reference

print(filecmp.cmp(f1, f2, shallow=False))
