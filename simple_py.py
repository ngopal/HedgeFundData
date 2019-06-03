import sys, os


data = sys.argv[1:]

print('\', \''.join([i.strip().replace(' ', '') for i in data]))
