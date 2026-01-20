import argparse
import json
import os
import random
from io import BytesIO

import pandas as pd
import matplotlib.pyplot as plt
import textwrap

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from PyPDF2 import PdfReader, PdfWriter
from os import mkdir

from config import (
    DB_PATH,
    BENCHMARK_FILE
)

def evaluation(selected):
    cwd = os.getcwd()
    writer = PdfWriter()
    for id in selected:

        doc = SimpleDocTemplate(os.path.join(cwd, "eval", f"evaluation_{id}", f"report_{id}.pdf"), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph(f"Report of ID {id}", styles["Title"]))

        results = list()
        for i in range(0, args.m):

            with open(os.path.join(cwd,"eval",f"evaluation_{id}",f"{id}_{i}.json")) as f:
                data = json.load(f)
                results.append(data)

        data = []
        header= ["Name", "Query", "EA", "SA" ]
        data.append(header)
        row = ["NL Query",Paragraph(results[0]['nl_querry'], styles['Normal']), "-", "-" ]
        data.append(row)
        row = [ "Gold Query", Paragraph(results[0]['gold_querry'], styles['Normal']), "-", "-"]
        data.append(row)

        for i,r in enumerate(results):
            row =[f"LLM Queray #{i}",Paragraph(r['llm_querry'], styles['Normal']), r['jaccard'], r['spider'] ]
            data.append(row)

        table = Table(data, colWidths=[3.7*cm , 12*cm,1*cm,1*cm])
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),  # header background
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # header text color
            ('GRID', (0, 0), (-1, -1), 1, colors.black),  # table grid lines
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),  # align text at top of cells
            ('LEFTPADDING', (0, 0), (-1, -1), 6),  # padding inside cells
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ])
        story.append(table)
        story.append(Spacer(1, 12))
        story.append(Paragraph("(Requests; llm query):", styles['Normal']))
        # Apply style to table
        table.setStyle(style)


        n_values = list()
        times = list()
        for i, r in enumerate(results):
            n_values.append(i)
            times.append(r['llm_requests'])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(n_values, times, color='skyblue')
        ax.xaxis.set_major_locator(FixedLocator(n_values))
        ax.set_xlabel("LLM Query number")
        ax.set_ylabel("requests")

        # Save plot to in-memory buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Add plot to story
        story.append(Image(buf, width=400, height=200))  # adjust size as needed
        story.append(Spacer(1, 12))

        doc.build(story)



### AUTOMERGE TO BIG REPORT
        reader = PdfReader(os.path.join(cwd, "eval", f"evaluation_{id}", f"report_{id}.pdf"))
        for page in reader.pages:
            writer.add_page(page)
        pass
    with open(os.path.join(cwd, "eval", f"report.pdf"), "wb") as f:
        writer.write(f)
    return 0

def data_generation(selected):
    # Create folder structure for generated 10*n files
    os.system('mkdir eval')
    for id in selected:
        print(f"Start evaluation of {id}:")
        os.system(f'mkdir '+os.path.join("eval",f"evaluation_{id}"))

        for i in range(0,args.m):
            #os.system(f"touch "+os.path.join("eval",f"evaluation_{id}",f"{id}_{i}"))
            os.system(f'python3 query_workflow.py --id {id} --save '+os.path.join("eval",f"evaluation_{id}",f"{id}_{i}"))
    return 0

def start():
    parent_dir = os.getcwd()
    #ToDo Ugly solution should be changed
    path= os.path.join(parent_dir,'..', DB_PATH, BENCHMARK_FILE)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    item_with_highest_id = max(data, key=lambda x: x["question_id"], default=None)
    print(f"ID: {item_with_highest_id["question_id"]}")
    highest_id = item_with_highest_id["question_id"]
    if args.n > highest_id:
        print("to many questions selected")
        return 1
    selected = random.sample(range(0,highest_id+1), args.n)
    #selected = [513,1483]
    data_generation(selected)
    evaluation(selected)

    return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automated evaluation.")
    parser.add_argument("--n", type=int, default=20, help="amount of arbitrary choosen questions")
    parser.add_argument("--m", type=int, default=DB_PATH, help="avoid divergence in model by multiple executions")
    args = parser.parse_args()
    start()