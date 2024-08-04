from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline

#function to create preprocessing pipeline 
def create_preprocessing_pipeline():
    num_cols = make_column_selector(dtype_include='number')
    cat_cols = make_column_selector(dtype_include='object')

    num_pipe = Pipeline([
        ('imputer', KNNImputer()),
         ('scaler', StandardScaler())
])
    cat_pipe = Pipeline([
        ('encoding', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

    preprocessor = ColumnTransformer([
        ('numeric', num_pipe, num_cols),
        ('categorical', cat_pipe, cat_cols)
        ])

    return preprocessor