from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from classifier_test_framework import ClassifierTestFramework
import pandas as pd

df = pd.read_csv("graduation_train.csv")

# print(df.isnull().sum())

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

for colum in df.columns:
    df = df[df[colum] != ""]

print(df)

df["curricular_units_1st_sem_grade_rounded"] = df["curricular_units_1st_sem_grade"].round(0)
df["curricular_units_2nd_sem_grade_rounded"] = df["curricular_units_2nd_sem_grade"].round(0)

colum_combinations = {
    "my_selection": [
        "curricular_units_1st_sem_grade_rounded", "curricular_units_1st_sem_evaluations",
        "curricular_units_1st_sem_approved",
        "curricular_units_2nd_sem_grade_rounded", "curricular_units_2nd_sem_evaluations",
        "curricular_units_2nd_sem_approved",
        "course", "previous_qualification", "special_needs"
    ],
    "academic_performance_indicators": [
        'curricular_units_1st_sem_credited',
        'curricular_units_1st_sem_enrolled', 'curricular_units_1st_sem_evaluations',
        'curricular_units_1st_sem_approved', 'curricular_units_1st_sem_grade',
        'curricular_units_1st_sem_without_evaluations', 'curricular_units_2nd_sem_credited',
        'curricular_units_2nd_sem_enrolled', 'curricular_units_2nd_sem_evaluations',
        'curricular_units_2nd_sem_approved', 'curricular_units_2nd_sem_grade',
        'curricular_units_2nd_sem_without_evaluations',
        'curricular_units_1st_sem_grade_rounded', 'curricular_units_2nd_sem_grade_rounded'
    ],
    "person_and_economic_indicator": [
        'marital_status', 'application_mode', 'application_order', 'course', 'attendance_type',
        'previous_qualification', 'nationality', 'mother_qualification', 'father_qualification', 'mother_occupation',
        'father_occupation', 'displaced', 'special_needs', 'debtor', 'tuition_fees_up_to_date',
        'scholarship_holder', 'age_at_enrollment', 'international', 'curricular_units_1st_sem_credited',
        'unemployment_rate', 'inflation_rate', 'gdp'
    ],
    "person_and_economic_and_academic_performance_indicators": [
        'marital_status', 'application_mode', 'application_order', 'course', 'attendance_type',
        'previous_qualification', 'nationality', 'mother_qualification', 'father_qualification', 'mother_occupation',
        'father_occupation', 'displaced', 'special_needs', 'debtor', 'tuition_fees_up_to_date',
        'scholarship_holder', 'age_at_enrollment', 'international', 'curricular_units_1st_sem_credited',
        'curricular_units_1st_sem_enrolled', 'curricular_units_1st_sem_evaluations',
        'curricular_units_1st_sem_approved', 'curricular_units_1st_sem_grade',
        'curricular_units_1st_sem_without_evaluations', 'curricular_units_2nd_sem_credited',
        'curricular_units_2nd_sem_enrolled', 'curricular_units_2nd_sem_evaluations',
        'curricular_units_2nd_sem_approved', 'curricular_units_2nd_sem_grade',
        'curricular_units_2nd_sem_without_evaluations', 'unemployment_rate', 'inflation_rate', 'gdp',
        'curricular_units_1st_sem_grade_rounded', 'curricular_units_2nd_sem_grade_rounded'
    ]
}

classifier_test_framework = ClassifierTestFramework(df, colum_combinations)

classifier_test_framework.add_classifier("DecisionTree", tree.DecisionTreeClassifier())
for i in range(1,20,5):
    classifier_test_framework.add_classifier(f"knn_{i}", Pipeline(
        steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=i))]
    ))
classifier_test_framework.add_classifier("Naive_bayes", GaussianNB())
for i in range(1,50,5):
    classifier_test_framework.add_classifier(f"Random_forest_{i}", RandomForestClassifier())


# for i in [5, 10, 15]:
#     classifier_test_framework.add_classifier(f"RF_SelectKBest_{i}", Pipeline(
#         steps=[
#             ("select", SelectKBest(score_func=f_classif, k=10)),
#             ("clf", RandomForestClassifier(n_estimators=10))
#         ]
#     ))

for key, items in classifier_test_framework.get_results().items():
    print(f"==={key}===")
    for columns, results in items.items():
        print(f"---{columns}---")
        print(results["report"])
        print("-----")
    print("========")
classifier_test_framework.get_report()
