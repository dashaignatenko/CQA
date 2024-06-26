from openai import OpenAI
import pandas as pd
import json
import click
from tqdm import tqdm


@click.command()
@click.option("--key", required=True)
@click.option("--input-filepath", required=True)
@click.option("--output-filepath", required=True)
@click.option("--model", default="gpt-3.5-turbo")
@click.option("--num-regens", default=3)
def main(key: str, input_filepath: str, output_filepath: str, model: str, num_regens: int):
  client = OpenAI(api_key=key)

  prompts_df = pd.read_json(input_filepath)
  
  completions_dict = {}
  for index, row in tqdm(prompts_df.iterrows()):
    if row["Aspect"] == "":
      # content = ' '.join([f'You are an analyst, and you need to write a 300-word comparison of "{row["object1"]}" and "{row["object2"]}".',
      #             'You need to single out the better of the two.',
      #             'Please be as short and concise as possible but prioritize quality.',
      #             'Below is a list of related arguments for or against, please use only the relevant ones.',
      #             'When you use an argument, please cite its number in square brackets right after the usage.',
      #             'Below the summary, list the argument numbers you used.',
      #             f'If an argument is not relevant to "{row["object1"]}" or "{row["object2"]}" at all, do not use it.',
      #             'The needed structure is: summary (100 words),',
      #             'bullet-point list of the most topical avenues of comparison (200 words or more),',
      #             'the best option (3 words only).',
      #             'Please avoid artificial balancing of arguments.',
      #             '\nArgument list:',
      #             f'{row["arguments"]}'])
      # content = ' '.join([f'You are an analyst, and you need to write a 300-word comparison of {row["object1"]} and {row["object2"]}. ',
      #             f'Task: You need to single out the better of the two. ',
      #             'Requirements: ',
      #             '- be short and concise ',
      #             '- prioritize quality ',
      #             '- analyze the list of arguments below ',
      #             '- pick relevant ones ',
      #             '- you need at least 10 arguments, create your own if needed (add a [generated] tag) ',
      #             '- cite used argument numbers in brackets right after the usage ',
      #             '- below everything, write a list of argument numbers you used ',
      #             'Needed structure: ',
      #             '- summary (100 words) ',
      #             '- bullet-point list of aspects of comparison (200 words or more) ',
      #             '- the best option (3 words only) ',
      #             '- list of used arguments ',
      #             '\nArgument list: ',
      #             f'{row["arguments"]}'])
      # 28.03:
      content = ' '.join([
                  f'You are an analyst, write a 300-word comparison of {row["object1"]} and {row["object2"]}. ',
                  'Task: compare and choose the better of the two. ',
                  'Requirements: ',
                  '- be concise ',
                  '- analyze the list of arguments below ',
                  '- pick relevant ones ',
                  '- rephrase in your own words ',
                  '- cite used argument numbers in square brackets right after the usage ',
                  '- the summary needs to have 15 arguments, create some if needed (add a [generated] tag) ',
                  'Needed structure: ',
                  '- summary (100 words) ',
                  '- bullet-point list of main aspects of comparison (200 words or more) ',
                  '- the best option (1 word) ',
                  '- numbered list of used arguments ',
                  '\nArgument list: ',
                  f'{row["arguments"]}'
                  ])
    else:
      # content = ' '.join([f'You are an analyst, and you need to write a 300-word comparison of "{row["object1"]}" and "{row["object2"]}".',
      #             f'Please focus on {row["Aspect"]}.',
      #             'You need to single out the better of the two.',
      #             'Please be as short and concise as possible but prioritize quality.',
      #             'Below is a list of related arguments for or against, please use only the relevant ones.',
      #             'When you use an argument, please cite its number in square brackets right after the usage.',
      #             'Below the summary, list the argument numbers you used.',
      #             f'If an argument is not relevant to "{row["object1"]}" or "{row["object2"]}" at all, do not use it.',
      #             'The needed structure is: summary (100 words),',
      #             'bullet-point list of the most topical avenues of comparison (200 words or more),',
      #             'the best option (3 words only).',
      #             'Please avoid artificial balancing of arguments.',
      #             '\nArgument list:',
      #             f'{row["arguments"]}'])
      # content = ' '.join([f'You are an analyst, and you need to write a 300-word comparison of {row["object1"]} and {row["object2"]}. ',
      #             'Task: You need to single out the better of the two. ',
      #             f'Focus on {row["Aspect"]}. ',
      #             'Requirements: ',
      #             '- be short and concise ',
      #             '- prioritize quality ',
      #             '- analyze the list of arguments below ',
      #             '- pick relevant ones ',
      #             '- you need at least 10 arguments, create your own if needed (add a [generated] tag) ',
      #             '- cite used argument numbers in brackets right after the usage ',
      #             '- below everything, write a list of argument numbers you used ',
      #             'Needed structure: ',
      #             '- summary (100 words) ',
      #             '- bullet-point list of aspects of comparison (200 words or more) ',
      #             '- the best option (3 words only) ',
      #             '- list of used arguments ',
      #             '\nArgument list: ',
      #             f'{row["arguments"]}'])
      # 28.03:
      content = ' '.join([
                  f'You are an analyst, write a 300-word comparison of {row["object1"]} and {row["object2"]}. ',
                  'Task: compare and choose the better of the two. ',
                  f'Focus on {row["Aspect"]}. ',
                  'Requirements: ',
                  '- be concise ',
                  '- analyze the list of arguments below ',
                  '- pick relevant ones ',
                  '- rephrase in your own words ',
                  '- cite used argument numbers in square brackets right after the usage ',
                  '- the summary needs to have 15 arguments, create some if needed (add a [generated] tag) ',
                  'Needed structure: ',
                  '- summary (100 words) ',
                  '- bullet-point list of main aspects of comparison (200 words or more) ',
                  '- the best option (1 word) ',
                  '- numbered list of used arguments ',
                  '\nArgument list: ',
                  f'{row["arguments"]}'
                  ])

    completion = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": "You are an expert analyst."},
        {"role": "user", "content": content}
      ], n=num_regens
    )

    completions_dict[index] = {
      "prompt": content, 
      "completions": [completion.choices[i].message.content for i in range(num_regens)]
    }

  with open(output_filepath, 'w') as outfile:
    json.dump(completions_dict, outfile, indent=2)


if __name__ == "__main__":
  main()