import glob
import os
from tqdm import tqdm
from lxml import etree
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt


def get_all_text(element):
    """Extract all text from an XML element."""
    if element is not None:
        return ''.join(element.xpath('.//text()')).strip()
    return ""


def get_median_content(section_parts, section_content):
    """Find the median content of a section."""
    n = len(section_content)
    if n > 1:
        mid = n // 2
        if n % 2 == 0:
            return section_parts[mid-1:mid+1], section_content[mid-1:mid+1]
        else:
            return [section_parts[mid]], [section_content[mid]]
    return [], []


def parse_xml_file(file_path, namespace={'tei': 'http://www.tei-c.org/ns/1.0'}):
    """Parse XML file and extract median content and its parts."""
    paper_parts, paper_content = [], []
    section_counter = 0
    temp_section_parts, temp_section_content = [], []

    tree = etree.parse(file_path)
    root = tree.getroot()

    for div in root.findall('.//tei:text//tei:div', namespace):
        head = div.find('.//tei:head', namespace)
        if head is not None:
            median_parts, median_content = get_median_content(temp_section_parts, temp_section_content)
            paper_parts.extend(median_parts)
            paper_content.extend(median_content)
            section_counter += 1
            temp_section_parts, temp_section_content = [], []
        
        for p in div.findall('.//tei:p', namespace):
            text = get_all_text(p)
            identifier = f'section_{section_counter}_text_{len(temp_section_content) + 1}'
            temp_section_parts.append(identifier)
            temp_section_content.append(text)

    median_parts, median_content = get_median_content(temp_section_parts, temp_section_content)
    paper_parts.extend(median_parts)
    paper_content.extend(median_content)

    return paper_parts, paper_content


def get_papers_embeddings(directory_path, st_model):
    """Generate embeddings for papers in a directory."""
    papers = []
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    file_paths = glob.glob(os.path.join(directory_path, '**', '*.xml'), recursive=True)

    for path in tqdm(file_paths):
        paper_parts, paper_content = parse_xml_file(path, ns)
        paper_embs = st_model.encode(paper_content, convert_to_tensor=True, batch_size=512)
        papers.append({
            'topic': os.path.basename(path),
            'path': path,
            'paper_parts': paper_parts,
            'paper_content': paper_content,
            'paper_embs': torch.mean(paper_embs, axis=0, keepdim=True)
        })

    return papers


def calculate_cosine_similarity(papers):
    """Calculate cosine similarity between paper embeddings."""
    embeddings = torch.vstack([paper['paper_embs'] for paper in papers])
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
    return similarity_matrix


def visualize_cos_similarity_matrix(papers):
    """Visualize cosine similarity matrix using a heatmap."""

    similarity_matrix = calculate_cosine_similarity(papers)
    papers_topics = [paper['topic'] for paper in papers]

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(similarity_matrix.cpu().numpy(), annot=True, fmt=".2f",
                     cmap="YlGnBu", square=True, annot_kws={"size": 8},
                     xticklabels=papers_topics,
                 yticklabels=papers_topics)  # Adjust the annotation font size as needed

    plt.title('Science Papers Cosine Similarity ', size=16)  # Adjust title font size as needed
    plt.xlabel('Domain', size=14)  # Adjust label font size as needed
    plt.ylabel('Domain', size=14)  # Adjust label font size as needed

    # Improve layout to make room for labels and title
    plt.tight_layout()

    # Optionally, adjust the tick labels size if still necessary
    ax.set_xticklabels(ax.get_xticklabels(), size=10)  # Adjust tick size as needed
    ax.set_yticklabels(ax.get_yticklabels(), size=10)  # Adjust tick size as needed

    plt.show()
