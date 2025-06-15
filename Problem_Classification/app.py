import streamlit as st
import os
from mistralai import Mistral
from pydantic import BaseModel, Field
from typing import List
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import zipfile
import tempfile
from PIL import Image
import base64
from io import BytesIO

# 初始化Mistral客户端
api_key = "your_mistral_api_key"
client = Mistral(api_key=api_key)

os.environ["OPENAI_API_BASE"] = "your_openai_api_base_url"
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"


def process_pdf(uploaded_file, file_path, start_page, end_page):
    try:
        f_name = uploaded_file.name
        uploaded_pdf = client.files.upload(file={ "file_name": f_name, "content": open(file_path, "rb"),},purpose="ocr")

        retrieved_file = client.files.retrieve(file_id=uploaded_pdf.id)

        signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
        # 使用Mistral处理PDF
        api_key = "xQYVlswSAOzqKtZZj7Ia8hpT0ExBzTyP"
        new_client = Mistral(api_key=api_key)

        ocr_response = new_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
            include_image_base64=True,
        )

        # 下载pdf中的所有图片
        for ocrpage in ocr_response.pages:
            if ocrpage.images:
                for i in ocrpage.images:
                    base64_str = i.image_base64.split(',')[1]#

                    # 解码Base64字符串为二进制数据
                    image_data = base64.b64decode(base64_str)

                    # 将二进制数据转换为Image对象
                    image = Image.open(BytesIO(image_data))

                    # 显示图像
                    image.save(i.id)
    

        # 提取PDF中的题目
        class Problem(BaseModel):
            """Represents a problem extracted from the text. sometimes a problem has several subproblems"""
            problem_text: str = Field(..., description="The full text of the problem.")

        class ExtractionData(BaseModel):
            """Container for a list of extracted problems."""
            problems: List[Problem]

        prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are given a markdown text with multiple problems. You are a problem extracte expert. You just extract the problems from the text.
         Each problem starts with question number, like 3. xxx, 4. xxx
        and sometimes a question has several subquestions, so it starts with Question X-X X.xxx X.xxx. You also should extract the fig markdown expression.
        the output struture is List[str].
        Example output
        ['Quesion 1-2 refer to the xxxx [img-1.jpeg](img-1.jpeg) 1.xxxx, 2.xxxx', '3. A door is hinged on one side', ...] """),

        ("human", "{text}"),
    ]
)

        # 构建OpenAI model
        llm = ChatOpenAI(model="gpt-4o")

        # 具有structured output的llm
        structured_llm = llm.with_structured_output(ExtractionData, method="json_schema")

        # 构建chain
        chain = prompt | structured_llm

        # 构建函数，输入是chain以及对应的text，输出是一个问problem list，每个list元素是一个problem的txt
        def problem_extractor(designed_chain, text):
            output = designed_chain.invoke(text)
            return output.problems

        # 提取每一页的题目
        problem_list = []

        #for i in range(1,11):
        for i in range(start_page-1,end_page):
            p_list = problem_extractor(chain , ocr_response.pages[i].markdown)
            for p in p_list:
                problem_list.append(p)
        
        #st.write(problem_list[0])

        # 构建一个函数，输入是problem list，输出是一个md文件，把problem list中的问题写入到md文件中
        # 构建问题分类的chain，就是把问题分到不同的类别
        # 下面这个LLM，根据题目输出每个题目的类别
        class category(BaseModel):
            """Represents the category of a problem."""
            knowledge_point: str = Field(..., description="a category name of the problem")

        cl_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an expert in physics problems classification. You are given a text, and the text is a physics problem.
         You should classify the text into one of the following categories:
         1. Mechanics: The problem may be related to Newton's Laws, Motion, Energy, Work, Torque, Momentum, Elasticity, Inertia, Rotational Motion, Angular Momentum, and Center of Mass.
         2. Thermodynamics: The problem may be related to Heat, Thermodynamics, Energy, Heat Equation, Heat Work, Heat Transfer, Heat Capacity, and Thermodynamics of Isobaric and Isothermal Processes.
         3. Electricity and Magnetism:The problem may be related to Electricity, Magnetism, Electric Fields, Electric Potential, Electric Currents, Electric Resistivity, and Electric Circuits.
         4. Modern Physics: The problem may be related to Relativity, Quantum/Black‐body, Dark Matter, Radioactivity, History
         5. Astronomy & Gravitation: The problem may be related to Gravitation, Astronomy, Planetary Motion, Orbital Mechanics, photoelectric effect and Gravitational Waves.
         6. Fluids & Buoyancy: The problem may be related to Fluid Mechanics, Fluid Dynamics, Buoyancy, and Fluid Flow.
         7. Waves and optics: The problem may be related to Waves, Optics, and Light Propagation.
         8. Unit Conversion:The problem may be related to Unit Conversion, Prefixes, and SI Units.

         You should only output the category name(for example you think the problem is related to Newton's Laws, you should output "Mechanics"), and nothing else. """),

        ("human", "{text}"),
    ]
)

        # 初始化OpenAI model
        cl_llm = ChatOpenAI(model="gpt-4o")

        # 构建输出题目类型llm
        cl_llm = cl_llm.with_structured_output(category, method="json_schema")

        # 构建输出题目类型的chain
        cl_chain = cl_prompt | cl_llm

        # 构建一个字典，每个key是分类名称，每个value是一个list，list中是该分类的问题
        keys = ['Mechanics', 'Thermodynamics', 'Electricity and Magnetism',
        "Modern Physics", "Astronomy & Gravitation", "Fluids & Buoyancy", "Waves and optics", "Unit Conversion"]

        d_lists = {k: ["## "+ k] for k in keys}

        # 遍历problem_list，对每个问题进行分类  
        for txt in problem_list:
            cl_output = cl_chain.invoke(txt.problem_text)
            d_lists[cl_output.knowledge_point].append(txt.problem_text)

        #st.write(d_lists['Mechanics'])
        
        # 生成一个md文件，把d_lists中的内容写入到md文件中
        with open('questions.md', 'w', encoding='utf-8') as f:
            for k in keys:
                for line in d_lists[k]:
                    f.write(line+'<br>')
                    f.write('\n\n')

        # 这里可以添加你的处理逻辑，将OCR结果转换为Markdown
        #markdown_content = f  # 假设这是处理后的Markdown
        #st.write(markdown_content)

        return True
        
    except Exception as e:
        st.error(f"处理过程中发生错误: {e}")
        return None

def has_jpeg_images(directory="."):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            return True
    return False

def zip_jpeg_files(output_zip_name="jpeg_images.zip"):
    """
    将当前文件夹中的所有 JPEG 文件打包成一个 zip 文件

    Parameters:
    - output_zip_name: 压缩包文件名（默认 jpeg_images.zip）
    """
    jpeg_extensions = ('.jpg', '.jpeg')
    current_dir = os.getcwd()
    with zipfile.ZipFile(output_zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(current_dir):
            if filename.lower().endswith(jpeg_extensions) and os.path.isfile(filename):
                zipf.write(filename)
    #st.write(f"👋 zip file is ready！")
    return output_zip_name

def main():
    st.title("Problem Classification AI Tools")
 
    # 文件上传组件
    uploaded_file = st.file_uploader("上传PDF文件", type="pdf")

    if uploaded_file is not None:
        # 获取文件的字节内容
        file_bytes = uploaded_file.read()
    
        # 指定保存路径
        save_path = os.path.join("uploads", uploaded_file.name)
    
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
        # 将文件保存到磁盘
        with open(save_path, "wb") as f:
            f.write(file_bytes)
    
            st.success(f"文件已保存至: {save_path}")
    
        if uploaded_file is not None:
            start_page = st.number_input("Input the starting page:",
                                 min_value=1,
                                 max_value=100)
            end_page = st.number_input("Input the end page:",
                                 min_value=1,
                                 max_value=100)
    
    # 页面第一次加载时就初始化session_state
    if "markdown_ready" not in st.session_state:
        st.session_state.markdown_ready = False

    if "markdown_path" not in st.session_state:
        st.session_state.markdown_path = None

    if "zip_ready" not in st.session_state:
        st.session_state.zip_ready = False

    if "zip_path" not in st.session_state:
        st.session_state.zip_path = None
    
    # 改变session_state中的状态
    if uploaded_file is not None and st.button("Transform"):
        with st.spinner("正在处理PDF..."):
            if process_pdf(uploaded_file, save_path, start_page, end_page):
                st.session_state.markdown_ready = True
                st.session_state.markdown_path = "questions.md"

            is_img = has_jpeg_images(directory=".")
            if is_img:
                zip_path = zip_jpeg_files(output_zip_name="jpeg_images.zip")
                st.session_state.zip_ready = True
                st.session_state.zip_path = zip_path

    # 下载Markdown文件
    if st.session_state.markdown_ready and st.session_state.markdown_path:
        with open(st.session_state.markdown_path, "r", encoding="utf-8") as file:
            md_content = file.read()

        st.download_button(
            label="📥 Download Markdown File",
            data=md_content,
            file_name="question.md",
            mime="text/markdown")

    # 下载zip文件
    if st.session_state.zip_ready and st.session_state.zip_path:
        zip_path = zip_jpeg_files(output_zip_name="jpeg_images.zip")
       
        with open(st.session_state.zip_path, "rb") as f:
            st.download_button(
                label="📥 Download ZIP File",
                data=f,
                file_name="folder_backup.zip",
                mime="application/zip"
                )
            st.success("All Done!")


if __name__ == "__main__":
    main()