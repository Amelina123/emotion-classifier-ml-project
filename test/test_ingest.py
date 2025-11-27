import pandas as pd
from src.ingest import load_data, clean_data, save_data
from src.config import TEXT_COL, LABEL_COL

#test1
def test_load_data_return_df():
    df = load_data()
    assert isinstance(df, pd.DataFrame)

#test2
def test_load_data_return_columns():
    df = load_data()
    expected_columns = {TEXT_COL, LABEL_COL}
    assert len(df.columns) == 2
    assert expected_columns.issubset(df.columns)

#test3
def test_clean_data_remove_dublicates():
    df = pd.DataFrame({
        TEXT_COL: ["hello", "hello", "world"],
        LABEL_COL: [1, 1, 2]
    })
    cleaned = clean_data(df)    
    assert len(cleaned) == 2
    

#test4
def test_clean_data_normalizes_data():
    df = pd.DataFrame({
    TEXT_COL: ["hello   world", "  a   b c  "],
    LABEL_COL: [1, 2]
    })
    cleaned = clean_data(df)
    
    assert cleaned[TEXT_COL].tolist() == ["hello world", "a b c"]
    assert cleaned[LABEL_COL].dtype == int

#test5
def test_save_data_creates_file(tmp_path):
    df = pd.DataFrame({
        TEXT_COL: ["hello"],
        LABEL_COL: [1]
    })
    output_file = tmp_path / "out.csv"
    save_data(df, output_file)    
    assert output_file.exists()
    assert output_file.suffix == ".csv"  

#test6    
def test_save_data_creates_csv_file(tmp_path):
    df = pd.DataFrame({
        TEXT_COL: ["hello"],
        LABEL_COL: [1]
    })
    output_file = tmp_path / "out.csv"
    save_data(df, output_file)     
    loaded = pd.read_csv(output_file)
    assert loaded.equals(df)



