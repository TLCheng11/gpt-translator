from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks import get_openai_callback

load_dotenv()


def translate_text(target_language: str, source_text: str):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8)

    system_template = "You are a professional translator for a law firm, translate the input into {target_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{source_text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    with get_openai_callback() as cb:
        respond = chain.run(
            {"target_language": target_language, "source_text": source_text}
        )

        print(respond)

        print("input_char_count: ", len(source_text))
        print("input_token_count: ", cb.prompt_tokens)
        print("output_char_count: ", len(respond))
        print("output_token_count: ", cb.completion_tokens)
        print("total_cost: ", cb.total_cost)


if __name__ == "__main__":
    translate_text(
        "english",
        """
        一、基本概念

    　　签证是一国政府授权机关依照本国法律法规，为申请入、出或过境本国的外国人颁发的一种许可证明。

    　　根据国际法及国际惯例，任何一个主权国家，有权自主决定是否允许外国人入出其国（边）境，依照本国法律发给签证、拒发签证或吊销已经签发的签证。

    　　中国签证机关根据法律和相关规定，决定颁发签证的种类、次数、有效期和停留期，有权拒绝当事人的签证申请或吊销已经签发的签证。
        """,
    )
