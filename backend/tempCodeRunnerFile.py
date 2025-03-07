import os
diagrams = ["output/diagrams/"+f for f in os.listdir("output/diagrams") if f.endswith(".png")]    
print(diagrams)