import torch
from transformers import pipeline
import os

class DeepReleaseSystem:
    def __init__(self, summarizer_path, classifier_path):
        
        print("Đang tải Stage 1: Summarizer...")
        self.summarizer = pipeline(
            "summarization", 
            model=summarizer_path, 
            device=0 if torch.cuda.is_available() else -1
        )
        
        
        print("Đang tải Stage 2: Classifier...")
        self.classifier = pipeline(
            "text-classification", 
            model=classifier_path, 
            device=0 if torch.cuda.is_available() else -1
        )

    def generate_release_notes(self, pr_data_list, version="v1.0.0"):
        organized_notes = {
            "new-features": [],
            "fix-bugs": [],
            "documentation": [],
            "non-functional": []
        }

        print(f"\n--- Đang xử lý {len(pr_data_list)} PRs cho phiên bản {version} ---")

        for pr in pr_data_list:
            full_text = f"{pr['title']} [SEP] {pr['description']}"
            summary = self.summarizer(full_text, max_length=50, min_length=10)[0]['summary_text']
            category_result = self.classifier(summary)[0]
            label = category_result['label']
            if label in organized_notes:
                organized_notes[label].append(summary)
            else:
                organized_notes.setdefault("others", []).append(summary)

        return self._format_markdown(organized_notes, version)

    def _format_markdown(self, organized_notes, version):
        md = f"# Release Notes - {version}\n\n"

        display_names = {
            "new-features": "New Features",
            "fix-bugs": "Bug Fixes",
            "documentation": "Documentation",
            "non-functional": "System & Refactoring"
        }

        for key, display_name in display_names.items():
            if organized_notes.get(key):
                md += f"## {display_name}\n"
                for item in organized_notes[key]:
                    md += f"- {item}\n"
                md += "\n"
        
        return md

if __name__ == "__main__":
    SUM_MODEL = "E:/Code summerization/Deeprelease Reproduce/data/entry_summarizer_model_save/data_loader_final"
    CAT_MODEL = "E:/Code summerization/Deeprelease Reproduce/data/categories_model_save"

    deep_release = DeepReleaseSystem(SUM_MODEL, CAT_MODEL)


    test_prs = [
        {
            "title": "Fix mysqli warning on connection", 
            "description": "The mysqli warning testament next will return bool in PHP 8.1."
        },
        {
            "title": "Add yard documentation for faker", 
            "description": "Improved the developer documentation by adding yard docs to faker json."
        },
        {
            "title": "New auth module", 
            "description": "Implemented OAuth2.0 authentication for the core system."
        }
    ]

    final_md = deep_release.generate_release_notes(test_prs, version="v2.0-beta")
    
    with open("RELEASE_NOTES_DEMO.md", "w", encoding="utf-8") as f:
        f.write(final_md)
    
    print("\n Đã tạo xong file RELEASE_NOTES_DEMO.md!")