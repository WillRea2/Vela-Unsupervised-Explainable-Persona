import pandas as pd
from Prompts import data_explanation
from openai import OpenAI
import json
from pydantic import BaseModel

just_cluster = False

data_with_cluster = pd.read_csv('data_success.csv')

api_key="your_secret_key"

client = OpenAI(
    api_key=api_key
)

def results(data):
    # Exclude 'cluster_ID' and 'subcluster_ID' from calculations
    data_to_analyze = data.drop(columns=['cluster_ID', 'subcluster_ID'])

    # Function to calculate summary statistics for the dataset as a whole
    dataset_overall_stats = data_to_analyze.describe().transpose().drop('count', axis=1).round(3)

    # Organize grouped statistics by cluster and subcluster
    grouped_stats = {}

    # Group the data by 'cluster_ID'
    for cluster_id, cluster_group in data.groupby('cluster_ID'):
        # Drop 'cluster_ID' and 'subcluster_ID' for statistics and calculate for the whole cluster
        cluster_stats = (
            cluster_group
            .drop(columns=['cluster_ID', 'subcluster_ID'])
            .describe()
            .transpose()
            .drop('count', axis=1)
            .round(3)
            .to_dict()
        )

        # Add subcluster statistics for the current cluster
        subcluster_stats = {}
        for subcluster_id, subcluster_group in cluster_group.groupby('subcluster_ID'):
            # Drop 'cluster_ID' and 'subcluster_ID' for statistics
            subcluster_stats[subcluster_id] = (
                subcluster_group
                .drop(columns=['cluster_ID', 'subcluster_ID'])
                .describe()
                .transpose()
                .drop('count', axis=1)
                .round(3)
                .to_dict()
            )

        # Add cluster-level and subcluster-level statistics
        grouped_stats[cluster_id] = {
            'cluster_statistics': cluster_stats,
            'subcluster_statistics': subcluster_stats
        }

    # Calculate the difference between clusters/subclusters and overall statistics
    differences = {}

    # Add differences for each cluster as a whole
    for cluster_id, cluster_group in data.groupby('cluster_ID'):
        cluster_mean = cluster_group.drop(columns=['cluster_ID', 'subcluster_ID']).mean().round(3)
        overall_mean = data_to_analyze.mean().round(3)
        differences[cluster_id] = {
            'cluster_differences': {
                'difference_mean': (cluster_mean - overall_mean).round(3).to_dict(),
                'relative_diff_mean': ((cluster_mean - overall_mean) / overall_mean).round(3).to_dict() #overall_mean = 0 values go to nan
            }
        }

        # Add differences for each subcluster within the cluster
        subcluster_differences_from_cluster = {}
        subcluster_differences_from_overall = {}
        for subcluster_id, subcluster_group in cluster_group.groupby('subcluster_ID'):
            subcluster_mean = subcluster_group.drop(columns=['cluster_ID', 'subcluster_ID']).mean().round(3)
            subcluster_differences_from_cluster[subcluster_id] = {
                'difference_mean': (subcluster_mean - cluster_mean).round(3).to_dict(),
                'relative_diff_mean': ((subcluster_mean - cluster_mean) / cluster_mean).round(3).to_dict()
            }
            subcluster_differences_from_overall[subcluster_id] = {
                'difference_mean': (subcluster_mean - overall_mean).round(3).to_dict(),
                'relative_diff_mean': ((subcluster_mean - overall_mean) / overall_mean).round(3).to_dict()
            }


        # Add subcluster-level differences to the cluster
        differences[cluster_id]['subcluster_differences_from_cluster'] = subcluster_differences_from_cluster
        differences[cluster_id]['subcluster_differences_from_overall'] = subcluster_differences_from_overall

    # Organize results into a dictionary
    results = {
        'dataset_overall_statistics': dataset_overall_stats.to_dict(),
        'group_statistics': grouped_stats,
        'differences': differences
    }
    return results

# Pydantic model for structured outputs
class ClusterExplanation(BaseModel):
    short_description: str
    persona_explanation: str
    data_evidence: str

def explain_cluster_persona(cluster_name, cluster_level, cluster_data, data_explanation):
    """
    Explains each cluster or sub-cluster using ChatGPT API in the required format.
    """
    level_descriptor = "Cluster" if cluster_level == "cluster" else "Sub-cluster"
    if cluster_level == "cluster":
        system_prompt = (            "You are a data interpretation assistant. You will receive:\n"
            "- A descriptive text ('data_explanation') of what each numeric feature represents.\n"
            "- Data for all the clusters and subclusters and compartive statistics of and between them. In particular:\n\n"
            " 'overall_statistics': statistics for the dataset as a whole, you can compare your cluster to this\n"
            " 'cluster_statistics': statisitcs for the specific cluster you are analysing,"
            "'cluster_differences': differences in mean between the cluster and the overall dataset, as well as relative_diff_mean gives the relative difference in mean to the population as a whole and is therefore an important category, it is the cluster mean minus the overall mean divded by overall mean \n"
            "Your job:\n \n"
            "1. Produce a three-part JSON object describing this cluster:\n"
            "   - 'short_description': A few words summarizing the key persona traits that make it different then the rest.\n"
            "   - 'persona_explanation': A detailed explanation of the founders' persona without including any numbers. Keep it short and concise.\n"
            "   - 'data_evidence': Precise, data-backed evidence written in words, but also use numbers.\n\n"
            "Make sure the explanation is consistent with the numeric data and feature meanings. Do not try to be nice, be fatual and precise. \n"
            "There is no need to include the word 'founder', be concise and to the point."
        )
        user_prompt = (
            f"data_explanation:\n{data_explanation}\n\n"
            f"{level_descriptor} Name: {cluster_name}\n\n"
            f"Below is the statistics of the clusters and dataset:\n"
            f"{cluster_data}\n\n"
            "Please provide the output in this JSON format:\n"
            "{\n"
            "  \"short_description\": \"...\",\n"
            "  \"persona_explanation\": \"...\",\n"
            "  \"data_evidence\": \"...\"\n"
            "}\n"
        )
    else:
        system_prompt = (
            "You are a data interpretation assistant. You will receive:\n"
            "- A descriptive text ('data_explanation') of what each numeric feature represents.\n"
            "- Data for all the clusters and subclusters and compartive statistics of and between them. In particular:\n\n"
            " 'overall_statistics': statistics for the dataset as a whole, you can compare your cluster to this\n"
            " 'subcluster_statistics': statistics for the specific subcluster you are analysing,"
            "'subcluster_differences_from_overall': differences in mean between the subcluster and the overall dataset, as well as relative_diff_mean gives the relative difference in mean to the population as a whole \n"
            "'subcluster_differences_from_cluster': differences in mean between the subcluster and the cluster it belongs to, as well as relative_diff_mean gives the relative difference in mean to the cluster as a whole and is therefore an important category, it is the subcluster mean minus the cluster mean divded by cluster mean \n"
            "Your job:\n \n"
            "1. Produce a three-part JSON object describing this cluster:\n"
            "   - 'short_description': A few words summarizing the key persona traits that make it different then the rest.\n"
            "   - 'persona_explanation': A detailed explanation of the founders' persona without including any numbers. Keep it short and concise.\n"
            "   - 'data_evidence': Precise, data-backed evidence written in words, but also use numbers.\n\n"
            "Make sure the explanation is consistent with the numeric data and feature meanings. Do not try to be nice, be fatual and precise. \n"
            "There is no need to include the word 'founder', be concise and to the point."
        )
        user_prompt = (
            f"data_explanation:\n{data_explanation}\n\n"
            f"{level_descriptor} Name: {cluster_name}\n\n"
            f"Below is the statistics of the clusters:\n"
            f"{cluster_data}\n\n"
            "Please provide the output in this JSON format:\n"
            "{\n"
            "  \"short_description\": \"...\",\n"
            "  \"persona_explanation\": \"...\",\n"
            "  \"data_evidence\": \"...\"\n"
            "}\n"
        )
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=ClusterExplanation,
        temperature=0.0
    )
    return response.choices[0].message.content

def refine_cluster_descriptions(cluster_explanations, cluster_data, data_explanation):
    """
    Calls the API again with cluster descriptions and data to refine the descriptions for accuracy.
    The parameter 'cluster_data' should include stats for *all* clusters.
    """
    refined_explanations = {}
    for cluster_name, explanation in cluster_explanations.items():
        cluster_summary = cluster_data[cluster_name]
        
        system_prompt = (
            "You are an assistant tasked with refining cluster descriptions.\n"
            "You will receive:\n"
            "  - 'data_explanation': meaning of each numeric feature.\n"
            "  - The existing description of this cluster.\n"
            "  - The existing descriptions for *all* clusters.\n"
            "  - Numeric data summarizing *all* clusters.\n"
            "  - Numeric data for this specific cluster.\n\n"
            "Your job:\n"
            "1. Review the existing description and determine if anything was missed, is inaccurate, or could be improved.\n"
            "2. Update the 'short_description', 'persona_explanation', and 'data_evidence' fields for improved accuracy. Especially convert the numbers in the data_evidence into the meaning of the numbers using your own reasoning and the data_explanation.\n"
            "3. Return the refined explanation in JSON format.\n"
            "The criteria for the fields are as follows: "   
            " 'short_description': One or two or three words summarizing the key persona traits that make it different then the rest.\n"
            " 'persona_explanation': An explanation of the founders' persona without including any numbers. Keep it short and concise. Just a couple sentences\n"
            "  'data_evidence': Precise, data-backed evidence written in a short list.  When using numbers, try use the description of the category and describe what the numbers are actually telling you, for example instead of average education level of 2, look at the data explanation and say most commonly obtaining a masters degree. For values between 0 and 1, use percentages as the outcome. Also only use two significant figures.\n\n"
            "Make sure the explanation is consistent with the numeric data and feature meanings. Do not try to be nice, be fatual and precise."
        )

        user_prompt = (
            f"data_explanation:\n{data_explanation}\n\n"
            f"Existing Description for {cluster_name}:\n"
            f"{json.dumps(explanation, indent=2)}\n\n"
            f"Existing Descriptions for All Clusters:\n"
            f"{json.dumps(cluster_explanations, indent=2)}\n\n"
            f"All Cluster Data:\n{json.dumps(cluster_data, indent=2)}\n\n"
            f"This Cluster's Data:\n{json.dumps(cluster_summary, indent=2)}\n\n"
            "Please return your refined JSON:\n"
            "{\n"
            "  \"short_description\": \"...\",\n"
            "  \"persona_explanation\": \"...\",\n"
            "  \"data_evidence\": \"...\"\n"
            "}\n"
        )

        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=ClusterExplanation,
            temperature=0.0
        )
        refined_explanations[cluster_name] = response.choices[0].message.content
    return refined_explanations

def refine_subcluster_descriptions(
    subcluster_explanations, 
    subcluster_name, 
    subcluster_data, 
    data_explanation
):
    """
    Calls the API again with sub-cluster descriptions and data to refine
    the descriptions for accuracy. Returns a JSON string containing the
    refined sub-cluster explanation.

    Parameters
    ----------
    subcluster_explanations : dict
        Dictionary of *all* sub-cluster explanations for the same parent cluster.
    subcluster_name : str
        Name of the sub-cluster in the dictionary (e.g. "Cluster 1, Sub-cluster 2").
    subcluster_data : dict
        Numeric data for this specific sub-cluster, plus 'all_subcluster_statistics'
        for the entire cluster.
    data_explanation : str
        Text describing the meaning of each numeric feature.
    """

    existing_explanation = subcluster_explanations[subcluster_name]

    system_prompt = (
        "You are an assistant tasked with refining cluster descriptions.\n"
        "You will receive:\n"
        "  - 'data_explanation': meaning of each numeric feature.\n"
#        "  - The existing sub-cluster description.\n"
        "  - Descriptions for all sub-clusters in the same parent cluster.\n"
        "  - Numeric data summarizing *all* sub-clusters in this cluster.\n"
#        "  - Numeric data for this specific sub-cluster.\n\n"
        f":Important: you are only refining the description for this sub-cluster, {subcluster_name}, therefore you must focus on this information, using the other information to compare it to.\n\n"
        "Your job:\n"
        "1. Review the existing description and determine if anything was missed, is inaccurate, or could be improved.\n"
        "2. Update the 'short_description', 'persona_explanation', and 'data_evidence' fields for improved accuracy. Especially convert the numbers in the data_evidence into the meaning of the numbers using your own reasoning and the data_explanation.\n"
        "3. Return the refined explanation in JSON format.\n"
        "The criteria for the fields are as follows: "   
        " 'short_description': One or two or three words summarizing the key persona traits that make it different then the rest.\n"
        " 'persona_explanation': An explanation of the founders' persona without including any numbers. Keep it short and concise. Just a couple sentences\n"
        "  'data_evidence': Precise, data-backed evidence written in a short list.  When using numbers, try use the description of the category and describe what the numbers are actually telling you, for example instead of average education level of 2, look at the data explanation and say most commonly obtaining a masters degree. For values between 0 and 1, use percentages as the outcome. Also only use two significant figures.\n\n"
        "Make sure the explanation is consistent with the numeric data and feature meanings. Do not try to be nice, be fatual and precise."
        "Make the subcluster descriptions in relation to the cluster descriptions and MAKE EACH SUBCLUSTER DESCRIPTION UNIQUE, YOUR TASK IS TO FIND THE DIFFERENCES BETWEEN THEM (but don't make anything up, thats still rule 1.)."
    )

    user_prompt = (
        f"data_explanation:\n{data_explanation}\n\n"
#        f"Existing Description for {subcluster_name}:\n"
#        f"{json.dumps(existing_explanation, indent=2)}\n\n"
        f"Existing Descriptions for All Subclusters in This Cluster:\n"
        f"{json.dumps(subcluster_explanations, indent=2)}\n\n"
        f"All Subcluster Data (parent cluster):\n"
        f"{json.dumps(subcluster_data['all_subcluster_statistics'], indent=2)}\n\n"
#        f"This Subcluster's Data:\n"
#        f"{json.dumps(subcluster_data, indent=2)}\n\n"
        "Please refine and return in JSON format:\n"
        "{\n"
        "  \"short_description\": \"...\",\n"
        "  \"persona_explanation\": \"...\",\n"
        "  \"data_evidence\": \"...\"\n"
        "}\n"
    )

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=ClusterExplanation,
        temperature=0.0
    )

    return response.choices[0].message.content


def output(data):
    # ------------------------------------------------------------------------------
    # PASS 1: INITIAL EXPLANATIONS
    # ------------------------------------------------------------------------------
    cluster_explanations = {}

    for cluster_id in data['cluster_ID'].unique():
        cluster_data_segment = data[data['cluster_ID'] == cluster_id]

        # Gather cluster-level stats
        dataset_overall_statistics = results['dataset_overall_statistics']
        cluster_statistics = results['group_statistics'][cluster_id]['cluster_statistics']
        cluster_differences = results['differences'][cluster_id]['cluster_differences']

        # Combine info for cluster-level explanation
        cluster_summary = {
            'dataset_overall_statistics': dataset_overall_statistics,
            'cluster_statistics': cluster_statistics,
            'cluster_differences': cluster_differences
        }

        # Generate explanation for the cluster
        cluster_key = f"Cluster {cluster_id}"
        cluster_explanation = explain_cluster_persona(
            cluster_name=cluster_key,
            cluster_level="cluster",
            cluster_data=cluster_summary,
            data_explanation=data_explanation
        )
        cluster_explanations[cluster_key] = json.loads(cluster_explanation)

        # If not just_cluster, we also do subclusters
        if not just_cluster:
            for subcluster_id in cluster_data_segment['subcluster_ID'].unique():
                subcluster_statistics = (
                    results['group_statistics'][cluster_id]['subcluster_statistics'][subcluster_id]
                )
                subcluster_differences_from_cluster = (
                    results['differences'][cluster_id]['subcluster_differences_from_cluster'][subcluster_id]
                )
                subcluster_differences_from_overall = (
                    results['differences'][cluster_id]['subcluster_differences_from_overall'][subcluster_id]
                )
                subcluster_key = f"Cluster {cluster_id}, Sub-cluster {subcluster_id}"

                subcluster_summary = {
                    'dataset_overall_statistics': dataset_overall_statistics,
                    'subcluster_statistics': subcluster_statistics,
                    'subcluster_differences_from_overall': subcluster_differences_from_overall,
                    'subcluster_differences_from_cluster': subcluster_differences_from_cluster
                }
                subcluster_explanation = explain_cluster_persona(
                    cluster_name=subcluster_key,
                    cluster_level="subcluster",
                    cluster_data=subcluster_summary,
                    data_explanation=data_explanation
                )
                cluster_explanations[subcluster_key] = json.loads(subcluster_explanation)

    # ------------------------------------------------------------------------------
    # PASS 2: REFINEMENT
    # ------------------------------------------------------------------------------
    # 1) Refine Clusters — Provide stats for *all* clusters
    all_clusters_data = {
        f"Cluster {cid}": data[data['cluster_ID'] == cid].describe().to_dict()
        for cid in data['cluster_ID'].unique()
    }

    # Separate cluster-level items
    just_cluster_explanations = {
        k: v for k, v in cluster_explanations.items() if "Sub-cluster" not in k
    }

    refined_main_cluster_explanations = refine_cluster_descriptions(
        cluster_explanations=just_cluster_explanations,
        cluster_data=all_clusters_data,
        data_explanation=data_explanation
    )

    # Merge refined cluster explanations
    refined_cluster_explanations = cluster_explanations.copy()
    for cluster_id in data['cluster_ID'].unique():
        ckey = f"Cluster {cluster_id}"
        refined_cluster_explanations[ckey] = refined_main_cluster_explanations[ckey]

    # 2) Refine Subclusters — Provide stats for *all* sub-clusters in the parent cluster
    for cluster_id in data['cluster_ID'].unique():
        cluster_key = f"Cluster {cluster_id}"
        cluster_data_segment = data[data['cluster_ID'] == cluster_id]

        # Gather all subcluster explanations for this cluster
        all_subcluster_explanations = {
            k: refined_cluster_explanations[k]
            for k in refined_cluster_explanations
            if k.startswith(cluster_key) and "Sub-cluster" in k
        }

        # For each subcluster in the cluster, refine
        for subcluster_id in cluster_data_segment['subcluster_ID'].unique():
            subcluster_key = f"Cluster {cluster_id}, Sub-cluster {subcluster_id}"
            
            # Provide stats for *all* subclusters in this parent cluster
            all_subcluster_statistics = (
                results['group_statistics'][cluster_id]['subcluster_statistics']
            )
            subcluster_statistics = all_subcluster_statistics[subcluster_id]
            subcluster_differences_from_cluster = (
                results['differences'][cluster_id]['subcluster_differences_from_cluster'][subcluster_id]
            )
            subcluster_differences_from_overall = (
                results['differences'][cluster_id]['subcluster_differences_from_overall'][subcluster_id]
            )

            subcluster_summary = {
                'all_subcluster_statistics': all_subcluster_statistics,
                'subcluster_statistics': subcluster_statistics,
                'subcluster_differences_from_cluster': subcluster_differences_from_cluster,
                'subcluster_differences_from_overall': subcluster_differences_from_overall
            }

            # Refine this specific subcluster, passing its key
            subcluster_explanation = refine_subcluster_descriptions(
                subcluster_explanations=all_subcluster_explanations,
                subcluster_name=subcluster_key,
                subcluster_data=subcluster_summary,
                data_explanation=data_explanation
            )
            # Update refined explanations
            refined_cluster_explanations[subcluster_key] = json.loads(subcluster_explanation)
    return refined_cluster_explanations

refined_cluster_explanations = output(data_with_cluster)

results = results(data_with_cluster)

# Save final refined output
output_file = "cluster_explanations_pass2.json"
with open(output_file, "w") as f:
    json.dump(refined_cluster_explanations, f, indent=2)

print("Second-pass cluster explanations saved to 'cluster_explanations_pass2.json'.")
print(json.dumps(refined_cluster_explanations, indent=2))
