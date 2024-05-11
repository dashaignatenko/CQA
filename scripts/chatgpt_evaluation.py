from openai import OpenAI
import pandas as pd
import click
from tqdm import tqdm


@click.command()
@click.option("--key", required=True)
@click.option("--input-filepath", required=True)
@click.option("--output-filepath", required=True)
@click.option("--model", default="gpt-4")
@click.option("--num-regens", default=1)
def main(key: str, input_filepath: str, output_filepath: str, model: str, num_regens: int):
    client = OpenAI(api_key=key)

    df = pd.read_json(input_filepath)

    completions_dict = {}
    for index in tqdm(range(len(df)), desc='Processing rows'):
        object1 = df.loc[index, 'object1']
        object2 = df.loc[index, 'object2']
        aspect = df.loc[index, 'aspect']
        aspect_text = f" when comparing regarding {aspect}" if aspect != '' else ''
        comparison = df.loc[index, "comparison"]
        content = ' '.join([
            'You are a helpful assistant.\n',
            'Task:\n',
            '- analyze the comparison given\n',
            '- for each criterion, assign points in the range given\n',
            'Criteria:\n',
            '1. a short introduction is present: 0-1\n',
            '   * the introduction is missing or is too long - 0 points\n',
            '   * the introduction is short and concise - 1 point\n',
            '2. there are defined aspects used for comparison: 0-1\n',
            '   * the comparison is arbitrary with no specific aspects - 0 points\n',
            '   * the summary uses specific aspects to compare objects - 1 point\n',
            '3. the introduction mentions the most important comparison aspects: 0-1\n',
            '   * no aspects are mentioned or no introduction - 0 points\n',
            '   * several most important aspects are mentioned in the introduction - 1 point\n',
            '4. the main body of the comparison has good structure: 0-1\n',
            '   * some aspects mix with others, the structure is harder to follow - 0 point\n',
            '   * the aspects are logically divided into separate aspects - 1 point\n',
            '5. the main body of the comparison has defined aspect names: 0-1\n',
            '   * no aspect names are given, comparison is inconcrete - 0 points\n',
            '   * main body has distinct aspect names - 1 point\n',
            '6. the main body of the comparison has defined aspect descriptions: 0-1\n',
            '   * no aspect descriptions are given, comparison is inconcrete - 0 points\n',
            '   * main body has distinct aspect descriptions - 1 point\n',
            '7. the final choice is given explicitly: 0-1\n',
            '   * no explicit choice made or lengthy justification present - 0 points\n',
            '   * short and explicit choice made - 1 point\n',
            '8. the comparison aspects in the main body of the comparison are sorted by general applicability: 0-1\n',
            '   * statements are not sorted at all - 0 points\n',
            '   * statements are sorted by general/important statements first, specific statements closer to the end - 1 point\n', 
            '9. each argument is relevant to the subject of comparison: 0-2\n',
            '   * most arguments are irrelevant - 0 points\n', 
            '   * most arguments are relevant - 1 point\n', 
            '   * all arguments are relevant - 2 points\n', 
            '10. each argument compares both objects: 0-2\n',
            '   * some arguments do not compare the objects - 0 points\n', 
            '   * some arguments give information only about one object - 1 points\n', 
            '   * all arguments compare both objects - 2 points\n', 
            '11. there are no hallucinations or statements contradicting common knowledge: 0-2\n',
            '   * many hallucinations, serious factual inaccuracy - 0 points\n',
            '   * some hallucinations, but mostly correct - 1 point\n',
            '   * no hallucinations, factually correct - 2 points\n',
            '12. the comparison has proper language and is easy to follow: 0-2\n',
            '   * hard to read, profanity present or illogical - 0 points\n',
            '   * some grammar issues, broken logic - 1 point\n',
            '   * no grammar issues, good structure and logic - 2 points \n',
            '13. there are no repetitive statements or statements too similar to each other: 0-1\n',
            '   * some statements repeat others’ meaning very closely - 0 points\n',
            '   * all statements are unique and do not repeat - 1 points\n',
            '14. the final answer is concluded from the statements in the main body (if all statements favor object 1, then the answer is object 1, if both objects are equally good or equally bad, then none of the objects is preferred and the answer is inconclusive): 0-1\n',
            '   * the final answer is not concluded from the arguments or no answer is given - 0 points\n',
            '   * the final answer is concluded from the majority of arguments - 1 point\n',
            '15. the summary itself is not too short and not too long: 0-1\n',
            '   * the summary is too short (less than 12 sentences) or too long (more than 20 sentences) - 0 points\n',
            '   * the summary is reasonably long (from 12 to 20 sentences) - 1 point\n',
            'Output a python dictionary with the structure: {"n": score, "n+1": score}\n',
            'Write only the dictionary, do not write anything else\n',
            f'Question: What is better{aspect_text}: {object1} or {object2}?\n',
            'Comparative answer:\n',
            f'{comparison}\n'
            ])
    
        completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert summary reviewer."},
            {"role": "user", "content": content}
        ], n=num_regens
        )

        completions_dict[index] = {
        "object1": object1, 
        "object2": object2, 
        "aspect": aspect, 
        "comparison": comparison, 
        "score_dict": completion.choices[0].message.content
        }
    
    out_df = pd.DataFrame(completions_dict)
    out_df.to_json(output_filepath, orient='records')


if __name__ == "__main__":
  main()