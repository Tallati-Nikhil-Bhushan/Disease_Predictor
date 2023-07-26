class Model:
    def __init__(self,title,description,result_yes,result_no,attributes):
        self.title = title
        self.description = description
        self.result_yes = result_yes
        self.result_no = result_no
        self.attributes = attributes


models = {
    "breast" : Model("Breast Cancer Predictor"
                    ,"Predict and prevent breast cancer with our innovative machine learning model. Get accurate risk assessments and take control of your health. Schedule now for peace of mind"
                    ,"The breast cancer is malignant"
                    ,"The breast cancer is benign"
                    ,['Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean',
       'Smoothness Mean', 'Compactness Mean', 'Concavity Mean',
       'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean',
       'Radius Standard Error', 'Texture Standard Error', 'Perimeter Standard Error', 'Area Standard Error', 'Smoothness Standard Error',
       'Compactness Standard Error', 'Concavity Standard Error', 'Concave Points Standard Error', 'Symmetry Standard Error',
       'Fractal Dimension Standard Error', 'Radius Worst', 'Texture Worst',
       'Perimeter Worst', 'Area Worst', 'Smoothness Worst',
       'Compactness Worst', 'Concavity Worst', 'Concave Points Worst',
       'Symmetry Worst', 'Fractal Dimension Worst']),

    "diabetes" : Model("Diabetes Predictor"
                      ,"Take control of your health with our advanced Diabetes Prediction Model. Accurately assess risk and make proactive decisions. Schedule now for a healthier future."
                      ,"Having Diabetes"
                      ,"Not Having Diabetes"
                      ,['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin',
       'BMI', 'Diabetes Pedigree Function', 'Age']),

    "heart" : Model("Heart Disease Predictor"
                          ,"Stay ahead of Heart Disease with our innovative prediction model. Accurately assess risk and take control of your health. Schedule now for a healthier future."
                          ,"Having Heart Disease"
                          ,"Not Having Heart Disease"
                          ,['Age', 'Gender', 'Chest Pain Type', 'Resting Blood Pressure', 'Serum Cholesterol in mg/dl', 'Fasting Blood Sugar 120 mg/dl', 'Resting Electrocardiographic Results', 'Maximum Heart Rate Achieved',
       'Exercise Induced Angina', 'ST Depression Induced by Exercise Relative to rest', 'Slope of the Peak Exercise ST segment', 'Number of Major vessels', 'thal']),

    
    "kidney" : Model("Kidney Disease Predictor"
                     ,"Protect your kidney health with our cutting-edge prediction model. Accurately assess risk and make proactive decisions. Schedule now for a healthier future."
                     ,"Having Kidney Disease"
                     ,"Not Having Kidney Disease"
                     ,['Age', 'Blood Pressure', 'Specific Gravity', 'Albumin', 'Sugar', 'Blood Glucose Random', 'Blood Urea', 'Serum Creatinine', 'Sodium', 'potassium', 'Hemoglobin',
       'Packed cell volume', 'White blood cell count', 'Red blood cell count', 'Red blood cells normal', 'Pus cell normal', 'Pus cell clumps present',
       'Bacteria present', 'Hypertension', 'Diabetes Mellitus', 'Coronary Artery Disease', 'Appetite Poor', 'Pedal Edema',
       'Anemia']),

    "liver" : Model("Liver Disease "
                    ,"Take charge of your liver health with our advanced prediction model. Accurately assess risk and make informed decisions. Schedule now for a healthier future."
                    ,"Having Liver Disease"
                    ,"Not Having Liver Disease"
                    ,['Age', 'Gender', 'Total Bilirubin in mg/dL', 'Conjugated Bilirubin in mg/dL',
       'Alkaline Phosphatase in IU/L', 'Alamine Aminotransferase in IU/L',
       'Aspartate Aminotransferase in IU/L', 'Total Proteins g/dL', 'Albumin g/dL',
       'Albumin and Globulin Ratio']),

    "stroke" : Model("Stroke Predictor"
                     ,"Prevent the unexpected with our innovative Stroke Prediction Model. Accurately assess risk and take control of your health. Schedule now for a brighter future."
                     ,"Having Chances Of Stroke"
                     ,"Not Having Chances Of Stroke"
                     ,['age', 'Hypertension', 'Heart disease', 'Average glucose level in blood', 'Body Mass Index',
       'Gender Male', 'Gender Other', 'Ever Married',
       'Never Worked', 'Work Type Private',
       'Work Type Self-employed', 'Are You Child', 'Residence type Urban',
       'Formerly smoked', 'Never smoked',
       'Smokes'])
}


categorical_columns = ['Fasting Blood Sugar 120 mg/dl','Exercise Induced Angina','Red blood cells normal', 'Pus cell normal', 'Pus cell clumps present','Bacteria present', 'Hypertension', 'Diabetes Mellitus', 'Coronary Artery Disease', 'Appetite Poor', 'Pedal Edema','Anemia',
'Hypertension', 'Heart disease','Gender Male', 'Gender Other', 'Ever Married',
       'Never Worked', 'Work Type Private',
       'Work Type Self-employed', 'Are You Child', 'Residence type Urban',
       'Formerly smoked', 'Never smoked',
       'Smokes']