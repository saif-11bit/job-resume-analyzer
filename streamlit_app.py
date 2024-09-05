import streamlit as st
import spacy
import re
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
from utils import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from collections import Counter
# load default skills data base
# import skill extractor
nlp = spacy.load("en_core_web_md")
# init skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def extract_skills(job_description, resume_text):
    extracted_job_soft_skill_names = []
    extracted_job_hard_skill_names = []
    extracted_resume_soft_skill_names = []
    extracted_resume_hard_skill_names = []
    jd_annotations = skill_extractor.annotate(job_description)
    resume_annotations = skill_extractor.annotate(resume_text)
    for i in jd_annotations["results"].keys():
        for skill in jd_annotations["results"][i]:
            skill_name = SKILL_DB[skill['skill_id']]['skill_name']
            skill_type = SKILL_DB[skill['skill_id']]['skill_type']
            if skill_type == "Hard Skill":
                extracted_job_hard_skill_names.append(skill_name)
            elif skill_type == "Soft Skill":
                extracted_job_soft_skill_names.append(skill_name)

    for i in resume_annotations["results"].keys():
        for skill in resume_annotations["results"][i]:
            skill_name = SKILL_DB[skill['skill_id']]['skill_name']
            skill_type = SKILL_DB[skill['skill_id']]['skill_type']
            if skill_type == "Hard Skill":
                extracted_resume_hard_skill_names.append(skill_name)
            elif skill_type == "Soft Skill":
                extracted_resume_soft_skill_names.append(skill_name)


    missing_soft_skills = list(set(extracted_job_soft_skill_names) - set(extracted_resume_soft_skill_names))
    matching_soft_skills = list(set(extracted_job_soft_skill_names) & set(extracted_resume_soft_skill_names))
    matching_hard_skills = list(set(extracted_job_hard_skill_names) & set(extracted_resume_hard_skill_names))
    missing_hard_skills = list(set(extracted_job_hard_skill_names) - set(extracted_resume_hard_skill_names))

    return missing_soft_skills, matching_soft_skills, matching_hard_skills, missing_hard_skills


def extract_ner(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in {'PRODUCT', 'LANGUAGE', 'ORG', 'SKILL'}]
    return entities

def extract_keywords(preprocessed_text):
    words = preprocessed_text.split()
    general_keywords = []
    entities = extract_ner(preprocessed_text)

    words = preprocessed_text.split()
    keywords = [word for word in words if len(word) > 1]
    keywords.extend(entities)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([' '.join(keywords)])
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    denselist = dense.tolist()
    tfidf_scores = {feature_names[i]: denselist[0][i] for i in range(len(feature_names))}

    sorted_keywords = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)
    for kw, score in sorted_keywords:
        general_keywords.append(kw)
    return general_keywords


def cosin_similarity_score(job_description, resume_text):
    similarity_score = util.pytorch_cos_sim(
        sentence_model.encode(job_description),
        sentence_model.encode(resume_text)
    ).item() * 100
    return similarity_score



def count_keyword_frequencies(keywords, job_description):
    # Normalize the job description
    
    # Create a Counter object to count word frequencies
    word_list = job_description.split()
    word_counts = Counter(word_list)
    
    keyword_frequencies = {}
    for keyword in keywords:
        if len(keyword) <= 2:
            continue
        # normalized_keyword = clean_text(keyword)
        if ' ' in keyword:
            # For multi-word keywords, use regex to count occurrences
            occurrences = len(re.findall(r'\b' + re.escape(keyword) + r'\b', job_description))
            keyword_frequencies[keyword] = occurrences
        else:
            # For single-word keywords, use the Counter object
            keyword_frequencies[keyword] = word_counts[keyword] if word_counts[keyword] > 0 else 1

    return keyword_frequencies

def main():
    st.title("Job Specific Analyzer")
    st.write("Welcome to my app!")

    # Job Description Input
    job_description = st.text_area("Enter the job description", "")

    job_description = clean_text(job_description)
    # Resume Text Input
    resume_text = st.text_area("Enter your resume text", "")

    # Submit Button
    if st.button("Submit"):
        missing_soft_skills, matching_soft_skills, matching_hard_skills, missing_hard_skills = extract_skills(job_description, resume_text)
        
        job_gn_keywords = extract_keywords(job_description)
        resume_gn_keywords = extract_keywords(resume_text)
        common_keywords = list(set(job_gn_keywords) & set(resume_gn_keywords))
        missing_keywords = list(set(job_gn_keywords) - set(resume_gn_keywords))[:15]

        missing_keywords_freq = count_keyword_frequencies(missing_keywords, job_description)
        common_keywords_freq = count_keyword_frequencies(common_keywords, job_description)

        similarity_score = cosin_similarity_score(job_description, resume_text)
        similarity_score = f"{similarity_score:.2f}"
        # Display the results
        st.write("Score:", similarity_score)
        
        st.write("Common Keywords:", common_keywords_freq)
        st.write("Missing Keywords:", missing_keywords_freq)
        st.write("Missing Soft Skills:", missing_soft_skills)
        st.write("Matching Soft Skills:", matching_soft_skills)
        st.write("Matching Hard Skills:", matching_hard_skills)
        st.write("Missing Hard Skills:", missing_hard_skills)
if __name__ == '__main__':
    main()
