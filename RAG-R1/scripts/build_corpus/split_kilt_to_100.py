import json

input_file_path = '../data/kilt/kilt_knowledgesource.json'
output_file_path = '../data/kilt/wiki_kilt_100_really.tsv'

idx = 0
line_num=0
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            line_num+=1
            if line_num% 100000==0:
                print("Finished Entity:",line_num)
                print("Now 100-words paragraph:",idx)
                print("=="*20)
            data = json.loads(line)

            text_list = data.get("text", [])
            title=data["wikipedia_title"]

            full_text = "".join(text_list)
            full_text = full_text.replace("BULLET::::","").replace("Section::::","")
            num_1=full_text.count("::::")
            num_2=full_text.count("print(a.split())")
            num_3=full_text.count("Section::::")
            if num_1!=num_2+num_3:
                print(full_text)
                print("=="*10)

            words = full_text.split()
            for i in range(0, len(words), 100):
                paragraph = " ".join(words[i:i + 100])
                paragraph = f"{idx}\t{title}   {paragraph}"
                output_file.write(f"{paragraph}\n")
                idx += 1

print(f"Finish，the results are available at {output_file_path}.")
