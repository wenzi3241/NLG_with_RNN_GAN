with open("final.txt", "r") as f:
	text = f.read()
	f.close()
sentences = text.split('\n')
print(len(sentences))
print(sentences)
output = []
for sentence in sentences:
	if sentence.startswith("Sheldon"):
		output.append(sentence)

with open("sheldon.txt", "w") as f:
	for sentence in output:
		f.write(sentence)
		f.write("\n")
	f.close()