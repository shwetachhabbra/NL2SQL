import streamlit as st
from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import load_dataset

# Load your fine-tuned model and tokenizer
CKPT = '/content/t5-small-finetuned-wikisql'
tokenizer = AutoTokenizer.from_pretrained(CKPT)
model = T5ForConditionalGeneration.from_pretrained(CKPT)

# Load the test data
test_data = load_dataset('wikisql', split='test')

def translate_to_sql(text):
    inputs = tokenizer(text, padding='longest', max_length=32, return_tensors='pt')
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=32)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit GUI
st.title("SQL Query Generation with T5 Model")

# Option to test with predefined queries or enter a custom query
option = st.selectbox(
    "Choose a query type",
    ("Simple Queries", "Multihop Queries", "Custom Query")
)

if option == "Simple Queries":
    st.header("Testing Simple Queries")

    simple_queries = [
        "What is terrence ross' nationality",
        "Which Frequency is used for WEGP calls?"
    ]

    for i, query in enumerate(simple_queries):
        st.subheader(f"Query {i + 1}: {query}")
        sql_prediction = translate_to_sql('translate to SQL: ' + query)
        expected_sql = test_data[i]['sql']['human_readable']
        
        st.write("**SQL Prediction:**")
        st.code(sql_prediction)
        st.write("**Expected SQL:**")
        st.code(expected_sql)

elif option == "Multihop Queries":
    st.header("Testing Multihop Queries")

    multihop_queries = [
        "How many players from the New Jersey Devils have a nationality that is also represented by a player in the Vancouver Canucks?",
        "List the names of wrestlers who had more than one reign and held a title for more than 100 days."
    ]

    multihop_expected_sql = [
        "SELECT COUNT(Player) FROM table WHERE NHL team = 'New Jersey Devils' AND Nationality IN (SELECT Nationality FROM table WHERE NHL team = 'Vancouver Canucks')",
        "SELECT Wrestler FROM table WHERE Number of reigns > 1 AND Days as champion > 100"
    ]

    for i, query in enumerate(multihop_queries):
        st.subheader(f"Query {i + 1}: {query}")
        sql_prediction = translate_to_sql('translate to SQL: ' + query)
        expected_sql = multihop_expected_sql[i]
        
        st.write("**SQL Prediction:**")
        st.code(sql_prediction)
        st.write("**Expected SQL:**")
        st.code(expected_sql)

elif option == "Custom Query":
    st.header("Enter a Custom Query")

    user_query = st.text_area("Input your query below:", height=100)

    if st.button("Generate SQL"):
        if user_query.strip() != "":
            sql_prediction = translate_to_sql('translate to SQL: ' + user_query)
            st.write("**SQL Prediction:**")
            st.code(sql_prediction)
        else:
            st.write("Please enter a valid query.")