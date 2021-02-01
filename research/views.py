import os
from socket import gethostbyname, gethostname
import numpy as np
import pickle as pkl
from django.shortcuts import render, HttpResponse

SHOW_VARIABLES = dict(
    online=gethostbyname(gethostname())!='127.0.0.1',
    offline=gethostbyname(gethostname())=='127.0.0.1'
)

# Create your views here.
def index(request):
    show_variables = dict(
        online=gethostbyname(gethostname())!='127.0.0.1',
        offline=gethostbyname(gethostname())=='127.0.0.1'
    )
    return render(request, 'research/index.html', context=show_variables)

def branch_prediction(request):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, '..', 'models', 'branch_prediction')
    encoder_dir = os.path.join(base_dir, '..', 'encoders', 'branch_prediction')
    scaler_dir = os.path.join(base_dir, '..', 'scalers', 'branch_prediction')
    unique_dir = os.path.join(base_dir, '..', 'unique', 'branch_prediction')
    data_dir = os.path.join(base_dir, '..', 'data', 'branch_prediction')
    
    # df = pd.read_csv(os.path.join(data_dir, 'STUDENT_DATA13.csv'))
    
    unique_college_names = pkl.load(open(f'{unique_dir}/college_names.sav','rb'))
    unique_college_codes = pkl.load(open(f'{unique_dir}/college_codes.sav','rb'))
    unique_genders = pkl.load(open(f'{unique_dir}/genders.sav','rb'))
    unique_candidate_types = pkl.load(open(f'{unique_dir}/candidate_types.sav','rb'))
    unique_categories = pkl.load(open(f'{unique_dir}/categories.sav','rb'))
    unique_home_universities = pkl.load(open(f'{unique_dir}/home_universities.sav','rb'))
    unique_ph_types = pkl.load(open(f'{unique_dir}/ph_types.sav','rb'))
    unique_defence_types = pkl.load(open(f'{unique_dir}/defence_types.sav','rb'))
    unique_nationalities = pkl.load(open(f'{unique_dir}/nationalities.sav','rb'))
    unique_cap_rounds = pkl.load(open(f'{unique_dir}/cap_rounds.sav','rb'))
    unique_branches = pkl.load(open(f'{unique_dir}/branches.sav','rb'))
    
    model = pkl.load(open(f'{model_dir}/XGBClassifier.sav','rb'))
        
    if request.method == 'POST':
        merit_no = request.POST['merit_no']
        merit_marks = request.POST['merit_marks']
        hsc_eligibility = request.POST['hsc_eligibility']
        college_name = request.POST['college_name']
        college_code = request.POST['college_code']
        gender = request.POST['gender']
        candidate_type = request.POST['candidate_type']
        category = request.POST['category']
        home_university = request.POST['home_university']
        ph_type = request.POST['ph_type']
        defence_type = request.POST['defence_type']
        nationality = request.POST['nationality']
        cap_round = request.POST['cap_round']
        # branch = request.POST['branch']
        
        gender_encoder = pkl.load(open(f'{encoder_dir}/Gender.sav','rb'))
        category_encoder = pkl.load(open(f'{encoder_dir}/Category.sav','rb'))
        candidate_type_encoder = pkl.load(open(f'{encoder_dir}/CandidateType.sav','rb'))
        college_name_encoder = pkl.load(open(f'{encoder_dir}/CollegeName.sav','rb'))
        nationality_encoder = pkl.load(open(f'{encoder_dir}/NATIONALITY.sav','rb'))
        defence_type_encoder = pkl.load(open(f'{encoder_dir}/DefenceType.sav','rb'))
        cap_round_encoder = pkl.load(open(f'{encoder_dir}/CAPRound.sav','rb'))
        ph_type_encoder = pkl.load(open(f'{encoder_dir}/PHType.sav','rb'))
        branch_encoder = pkl.load(open(f'{encoder_dir}/BRANCH.sav','rb'))
        home_university_encoder = pkl.load(open(f'{encoder_dir}/HomeUniversity.sav','rb'))
        college_code_encoder = pkl.load(open(f'{encoder_dir}/CollegeCode.sav','rb'))
        
        data = np.array([[
            college_name_encoder.transform([college_name]), college_code_encoder.transform([float(college_code)]), 
            merit_no, merit_marks, gender_encoder.transform([gender]), candidate_type_encoder.transform([candidate_type]), 
            category_encoder.transform([category]), home_university_encoder.transform([home_university]), 
            ph_type_encoder.transform([ph_type]), defence_type_encoder.transform([defence_type]), hsc_eligibility, 
            cap_round_encoder.transform([cap_round]), nationality_encoder.transform([nationality])
        ]])
        
        scaler = pkl.load(open(f'{scaler_dir}/scaler.sav', 'rb'))
        data = scaler.transform(data)
        
        init_results = np.argsort(model.predict_proba(data)[0]).tolist()[-3:]
        results = [branch_encoder.inverse_transform([init_results[i]]).tolist()[0] for i in range(2, -1, -1)]
        del init_results
        
        show_variables = dict(
            online=gethostbyname(gethostname())!='127.0.0.1',
            offline=gethostbyname(gethostname())=='127.0.0.1',
            results=results,
        )
        return render(request, 'research/branch_prediction_result.html', context=show_variables)
    
    show_variables = dict(
        online=gethostbyname(gethostname())!='127.0.0.1',offline=gethostbyname(gethostname())=='127.0.0.1',
        ucn=unique_college_names,
        ucc=unique_college_codes,
        ug=unique_genders,
        uct=unique_candidate_types,
        uc=unique_categories,
        uhu=unique_home_universities,
        upt=unique_ph_types,
        udt=unique_defence_types,
        un=unique_nationalities,
        ucr=unique_cap_rounds,
        ub=unique_branches,
    )
    return render(request, 'research/branch_prediction.html', context=show_variables)

def college_prediction(request):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, '..', 'models', 'college_prediction')
    encoder_dir = os.path.join(base_dir, '..', 'encoders', 'college_prediction')
    scaler_dir = os.path.join(base_dir, '..', 'scalers', 'college_prediction')
    unique_dir = os.path.join(base_dir, '..', 'unique', 'college_prediction')
    data_dir = os.path.join(base_dir, '..', 'data', 'college_prediction')
    
    unique_genders = pkl.load(open(f'{unique_dir}/genders.sav','rb'))
    unique_candidate_types = pkl.load(open(f'{unique_dir}/candidate_types.sav','rb'))
    unique_home_universities = pkl.load(open(f'{unique_dir}/home_universities.sav','rb'))
    unique_cap_rounds = pkl.load(open(f'{unique_dir}/cap_rounds.sav','rb'))
    unique_branches = pkl.load(open(f'{unique_dir}/branches.sav','rb'))
    unique_nationalities = pkl.load(open(f'{unique_dir}/nationalities.sav','rb'))
    unique_categories_ph_defence_types = pkl.load(open(f'{unique_dir}/category_ph_defence_types.sav','rb'))
    unique_college_names = pkl.load(open(f'{unique_dir}/college_names.sav','rb'))
    
    model = pkl.load(open(f'{model_dir}/XGBoostClassifier.sav','rb'))
    
    if request.method == 'POST':
        merit_no = request.POST['merit_no']
        merit_marks = request.POST['merit_marks']
        hsc_eligibility = request.POST['hsc_eligibility']
        gender = request.POST['gender']
        candidate_type = request.POST['candidate_type']
        home_university = request.POST['home_university']
        cap_round = request.POST['cap_round']
        branch = request.POST['branch']
        nationality = request.POST['nationality']
        category_ph_defence_type = request.POST['category_ph_defence_type']
        # college_name = request.POST['college_name']
        
        gender_encoder = pkl.load(open(f'{encoder_dir}/Gender.sav','rb'))
        candidate_type_encoder = pkl.load(open(f'{encoder_dir}/CandidateType.sav','rb'))
        home_university_encoder = pkl.load(open(f'{encoder_dir}/HomeUniversity.sav','rb'))
        cap_round_encoder = pkl.load(open(f'{encoder_dir}/CAPRound.sav','rb'))
        branch_encoder = pkl.load(open(f'{encoder_dir}/BRANCH.sav','rb'))
        nationality_encoder = pkl.load(open(f'{encoder_dir}/NATIONALITY.sav','rb'))
        category_ph_defence_encoder = pkl.load(open(f'{encoder_dir}/Category_PH_DefenceType.sav','rb'))
        college_name_encoder = pkl.load(open(f'{encoder_dir}/CollegeName.sav','rb'))
        
        data = np.array([[
            merit_no, merit_marks, gender_encoder.transform([gender]), candidate_type_encoder.transform([candidate_type]), 
            home_university_encoder.transform([home_university]), hsc_eligibility, cap_round_encoder.transform([cap_round]), 
            branch_encoder.transform([branch]), nationality_encoder.transform([nationality]), 
            category_ph_defence_encoder.transform([category_ph_defence_type])
        ]])
        
        scaler = pkl.load(open(f'{scaler_dir}/scaler.sav', 'rb'))
        data = scaler.transform(data)
        
        init_results = np.argsort(model.predict_proba(data)[0]).tolist()[-3:]
        results = [college_name_encoder.inverse_transform([init_results[i]]).tolist()[0] for i in range(2, -1, -1)]
        del init_results
        
        show_variables = dict(
            online=gethostbyname(gethostname())!='127.0.0.1',
            offline=gethostbyname(gethostname())=='127.0.0.1',
            results=results,
        )
        return render(request, 'research/college_prediction_result.html', context=show_variables)

    show_variables = dict(
        online=gethostbyname(gethostname())!='127.0.0.1',
        offline=gethostbyname(gethostname())=='127.0.0.1',
        ug=unique_genders,
        uct=unique_candidate_types,
        uhu=unique_home_universities,
        ucr=unique_cap_rounds,
        ub=unique_branches,
        un=unique_nationalities,
        ucpdt=unique_categories_ph_defence_types,
        ucn=unique_college_names,
    )
    return render(request, 'research/college_prediction.html', context=show_variables)

def placement_prediction(request):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, '..', 'models', 'placement_prediction')
    encoder_dir = os.path.join(base_dir, '..', 'encoders', 'placement_prediction')
    scaler_dir = os.path.join(base_dir, '..', 'scalers', 'placement_prediction')
    unique_dir = os.path.join(base_dir, '..', 'unique', 'placement_prediction')
    data_dir = os.path.join(base_dir, '..', 'data', 'placement_prediction')
    
    unique_branches = pkl.load(open(f'{unique_dir}/Branch.sav','rb'))
    unique_campus = pkl.load(open(f'{unique_dir}/Campus.sav','rb'))
    unique_genders = pkl.load(open(f'{unique_dir}/Gender.sav','rb'))
    
    model = pkl.load(open(f'{model_dir}/RandomForestClassifier.sav','rb'))
    
    if request.method == 'POST':
        branch = request.POST['branch']
        campus = request.POST['campus']
        gender = request.POST['gender']
        be_aggregate_marks = request.POST['be_aggregate_marks']
        semester1_marks = request.POST['semester1_marks']
        backpapers1 = request.POST['backpapers1']
        p_backpapers1 = request.POST['p_backpapers1']
        semester2_marks = request.POST['semester2_marks']
        backpapers2 = request.POST['backpapers2']
        p_backpapers2 = request.POST['p_backpapers2']
        semester3_marks = request.POST['semester3_marks']
        backpapers3 = request.POST['backpapers3']
        p_backpapers3 = request.POST['p_backpapers3']
        semester4_marks= request.POST['semester4_marks']
        backpapers4 = request.POST['backpapers4']
        p_backpapers4 = request.POST['p_backpapers4']
        semester5_marks = request.POST['semester5_marks']
        backpapers5 = request.POST['backpapers5']
        p_backpapers5 = request.POST['p_backpapers5']
        semester6_marks = request.POST['semester6_marks']
        backpapers6 = request.POST['backpapers6']
        p_backpapers6 = request.POST['p_backpapers6']
        semester7_marks = request.POST['semester7_marks']
        backpapers7 = request.POST['backpapers7']
        p_backpapers7 = request.POST['p_backpapers7']
        hsc_marks = request.POST['hsc_marks']
        ssc_marks = request.POST['ssc_marks']
        diploma_marks = request.POST['diploma_marks']
        dead_back_log = request.POST['dead_back_log']
        live_atkt = request.POST['live_atkt']
        
        branch_encoder = pkl.load(open(f'{encoder_dir}/Branch.sav','rb'))
        campus_encoder = pkl.load(open(f'{encoder_dir}/Campus.sav','rb'))
        gender_encoder = pkl.load(open(f'{encoder_dir}/Gender.sav','rb'))
        
        data = np.array([[
            branch_encoder.transform([branch]), campus_encoder.transform([campus]), gender_encoder.transform([gender]), 
            be_aggregate_marks, semester1_marks, backpapers1, p_backpapers1, 
            semester2_marks, backpapers2, p_backpapers2, semester3_marks, 
            backpapers3, p_backpapers3, semester4_marks, backpapers4, 
            p_backpapers4, semester5_marks, backpapers5, p_backpapers5, 
            semester6_marks, backpapers6, p_backpapers6, semester7_marks, 
            backpapers7, p_backpapers7, hsc_marks, ssc_marks, diploma_marks, 
            dead_back_log, live_atkt, 
        ]])
        # print(data)
        scaler = pkl.load(open(f'{scaler_dir}/scaler.sav', 'rb'))
        data = scaler.transform(data)
        
        init_results = model.predict(data).tolist()[0]
        if init_results == 0:
            results = 'Will Get Placement'
        elif init_results == 1:
            results = 'Wont Get Placement'
        else:
            results = 'Not Sure of Placement'
        print(results)
        
        show_variables = dict(
            online=gethostbyname(gethostname())!='127.0.0.1',
            offline=gethostbyname(gethostname())=='127.0.0.1',
            results=results,
        )
        return render(request, 'research/placement_prediction_result.html', context=show_variables)
    show_variables = dict(
        online=gethostbyname(gethostname())!='127.0.0.1',
        offline=gethostbyname(gethostname())=='127.0.0.1',
        ub=unique_branches,
        uc=unique_campus,
        ug=unique_genders,
    )
    return render(request, 'research/placement_prediction.html', context=show_variables)