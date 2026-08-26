from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"
with open(OUT_DIR / "results.json", encoding="utf-8") as f:
    r = json.load(f)

standard = r["standard"]
optimized = r["optimized"]

pdf_path = OUT_DIR / "relatorio_executivo_mloops_industria_4_0.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=20, leading=24, spaceAfter=14))
styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10.5, leading=15, textColor=colors.HexColor("#555555"), spaceAfter=18))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=15, leading=18, spaceBefore=8, spaceAfter=8, textColor=colors.HexColor("#16324F")))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=11.5, leading=14, spaceBefore=7, spaceAfter=5, textColor=colors.HexColor("#255A7B")))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9.4, leading=14, alignment=TA_JUSTIFY, spaceAfter=7))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=8.2, leading=11, textColor=colors.HexColor("#555555"), spaceAfter=4))
styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontSize=10, leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#16324F"), borderWidth=0.8, borderColor=colors.HexColor("#B7C9D6"), borderPadding=8, spaceBefore=8, spaceAfter=10))


def money(x: float) -> str:
    s = f"R$ {x:,.0f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float) -> str:
    return f"{x*100:.2f}%"


def p(text: str, style="Bodyx"):
    return Paragraph(text, styles[style])


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(1.7 * cm, 1.0 * cm, "Relatório Executivo - MLOps e Detecção de Falhas Industriais")
    canvas.drawRightString(width - 1.7 * cm, 1.0 * cm, f"Página {doc.page}")
    canvas.restoreState()


class Report(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, pagesize=A4, rightMargin=1.55*cm, leftMargin=1.55*cm, topMargin=1.55*cm, bottomMargin=1.55*cm, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates([PageTemplate(id="report", frames=frame, onPage=header_footer)])


story = []
story.append(Spacer(1, 1.0*cm))
story.append(p("Relatório Executivo", "TitleCenter"))
story.append(p("Detecção de Falhas Industriais com MLOps, Métricas para Classes Desbalanceadas e Ajuste Dinâmico de Threshold", "Subtitle"))
story.append(p("Indústria 4.0 | Dados sintéticos de sensores | 99,5% Operacional vs. 0,5% Falha", "Callout"))

summary_data = [
    ["Volume", f"{r['n_samples']:,}".replace(',', '.'), "Split", "70% treino / 15% validação / 15% teste"],
    ["Taxa de falha", pct(r["failure_rate_total"]), "Modelo", "Pipeline: StandardScaler + Logistic Regression balanceada"],
    ["Threshold padrão", "0,50", "Threshold otimizado", f"{optimized['threshold']:.2f}"],
    ["Custo padrão no teste", money(standard["total_cost"]), "Custo otimizado", money(optimized["total_cost"])],
    ["Economia", money(r["savings"]), "Redução", f"{r['savings_pct']:.2f}%"],
]
t = Table(summary_data, colWidths=[3.3*cm, 4.1*cm, 3.7*cm, 6.1*cm], hAlign="CENTER")
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#EAF1F6")),
    ("BACKGROUND", (2,0), (2,-1), colors.HexColor("#EAF1F6")),
    ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B8C5CF")),
    ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
    ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("TOPPADDING", (0,0), (-1,-1), 6),
]))
story.append(t)
story.append(Spacer(1, 0.35*cm))
story.append(p("Objetivo", "H1x"))
story.append(p("Construir uma solução reprodutível de Engenharia de Dados e MLOps para classificação de falhas raras em sensores industriais, demonstrando por que a acurácia não deve ser usada isoladamente, como a matriz de confusão altera a interpretação do modelo e como o threshold de decisão pode ser calibrado segundo o impacto operacional e financeiro."))

story.append(p("1. Arquitetura da Solução e Prevenção de Data Leakage", "H1x"))
story.append(p("O fluxo foi estruturado para preservar a independência estatística do conjunto de teste. A divisão estratificada ocorre antes do ajuste do pré-processamento: 70% dos registros formam o treinamento, 15% a validação e 15% o teste. O StandardScaler permanece dentro de um Pipeline do scikit-learn e é ajustado exclusivamente com X_train. A seleção do threshold econômico é feita sobre X_val, enquanto X_test permanece reservado para a mensuração final. Isso evita que médias, desvios-padrão, parâmetros do classificador ou decisões de threshold sejam influenciados pelo teste."))

arch = Table([
    ["Etapa", "Entrada", "Operação", "Saída"],
    ["1. Geração", "Distribuições sintéticas", "Sensores + variável-alvo desbalanceada", "100.000 registros"],
    ["2. Split", "Dataset completo", "Estratificação", "70% treino / 15% validação / 15% teste"],
    ["3. Pré-processamento", "X_train", "StandardScaler", "Parâmetros aprendidos só no treino"],
    ["4. Treinamento", "X_train + y_train", "Logistic Regression balanceada", "Modelo treinado"],
    ["5. Calibração", "X_val", "Varredura de thresholds + função de custo", f"Threshold = {optimized['threshold']:.2f}"],
    ["6. Avaliação", "X_test", "Métricas e matriz de confusão", "Resultado final não usado para tuning"],
], colWidths=[2.25*cm, 3.5*cm, 5.2*cm, 6.0*cm])
arch.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16324F")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#C1CCD4")),
    ("FONTSIZE", (0,0), (-1,-1), 7.7),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F6F8FA")]),
    ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
]))
story.append(arch)
story.append(p("Regra de ouro de MLOps: qualquer transformação que estime parâmetros a partir dos dados deve ser ajustada apenas em treino, e decisões de hiperparâmetro/threshold devem utilizar validação. O teste deve aparecer uma única vez no encerramento do ciclo de avaliação.", "Callout"))

story.append(p("2. Dados Sintéticos e Cenário de Desbalanceamento", "H1x"))
story.append(p("Foram sintetizados 100.000 registros com seis sinais industriais: vibração RMS, temperatura, pressão, corrente elétrica, pico de aceleração e ruído acústico. A classe Falha representa aproximadamente 0,5% dos eventos, conforme a situação proposta. Registros de falha recebem deslocamentos controlados nas distribuições dos sensores, mantendo sobreposição entre classes para evitar um problema artificialmente trivial."))
story.append(p("Nesse regime, um classificador ingênuo que previsse sempre Operacional teria acurácia próxima de 99,5%, mas recall de 0% para a classe crítica. Portanto, a pergunta operacional relevante não é apenas 'quantos acertamos?', mas principalmente 'quantas falhas conseguimos capturar e a que custo em falsos alarmes?'."))

story.append(p("3. Quadro Comparativo de Métricas", "H1x"))
headers = ["Métrica", "Threshold 0,50", f"Threshold {optimized['threshold']:.2f}"]
rows = [
    ["Acurácia", pct(standard["accuracy"]), pct(optimized["accuracy"])],
    ["TN", f"{standard['tn']:,}".replace(',', '.'), f"{optimized['tn']:,}".replace(',', '.')],
    ["FP", f"{standard['fp']:,}".replace(',', '.'), f"{optimized['fp']:,}".replace(',', '.')],
    ["FN", f"{standard['fn']:,}".replace(',', '.'), f"{optimized['fn']:,}".replace(',', '.')],
    ["TP", f"{standard['tp']:,}".replace(',', '.'), f"{optimized['tp']:,}".replace(',', '.')],
    ["Precisão", pct(standard["precision"]), pct(optimized["precision"])],
    ["Recall", pct(standard["recall"]), pct(optimized["recall"])],
    ["F1", pct(standard["f1"]), pct(optimized["f1"])],
    ["F2", pct(standard["f2"]), pct(optimized["f2"])],
    ["F0,5", pct(standard["f0_5"]), pct(optimized["f0_5"])],
    ["AUC-ROC", pct(standard["auc_roc"]), pct(optimized["auc_roc"])],
]
metric_table = Table([headers] + rows, colWidths=[6.1*cm, 5.2*cm, 5.2*cm], repeatRows=1)
metric_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16324F")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("ALIGN", (1,1), (-1,-1), "CENTER"),
    ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#C1CCD4")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F6F8FA")]),
    ("FONTSIZE", (0,0), (-1,-1), 8.2),
    ("TOPPADDING", (0,0), (-1,-1), 4.5), ("BOTTOMPADDING", (0,0), (-1,-1), 4.5),
]))
story.append(metric_table)
story.append(Spacer(1, 0.2*cm))
story.append(p("Leitura técnica: a AUC-ROC permaneceu em torno de 0,997, indicando excelente capacidade de ordenação das probabilidades. O threshold, porém, modifica a matriz de confusão e, consequentemente, Precisão, Recall e F-beta. O F2 pesa mais o Recall (beta > 1), sendo mais coerente com situações em que perder uma falha é muito mais grave do que gerar um alarme adicional. Já F0,5 pesa mais a Precisão."))

story.append(Image(str(FIG_DIR / "metric_comparison.png"), width=16.6*cm, height=8.8*cm))
story.append(Paragraph("Figura 1 - Comparação das métricas no conjunto de teste.", styles["Smallx"]))
story.append(PageBreak())

story.append(p("4. Interpretação Matemática das Métricas", "H1x"))
story.append(p("A matriz de confusão é definida por quatro contagens: <b>TP</b> (falha corretamente detectada), <b>TN</b> (operação corretamente identificada), <b>FP</b> (falso alarme) e <b>FN</b> (falha não detectada). A partir dessas quantidades, temos:"))
formula_data = [
    ["Acurácia", "(TP + TN) / (TP + TN + FP + FN)"],
    ["Precisão", "TP / (TP + FP)"],
    ["Recall", "TP / (TP + FN)"],
    ["F_beta", "(1 + beta²) * Precisão * Recall / (beta² * Precisão + Recall)"],
]
ft = Table(formula_data, colWidths=[4.0*cm, 12.5*cm])
ft.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#EAF1F6")),
    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
    ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#C1CCD4")),
    ("FONTSIZE", (0,0), (-1,-1), 8.8),
    ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
]))
story.append(ft)
story.append(p("Na classe rara, a Acurácia sofre de efeito de prevalência: milhares de TN podem dominar a razão mesmo quando o modelo tem desempenho ruim na classe Falha. Por isso, o relatório prioriza a matriz de confusão, Recall, Precisão, família F-beta e AUC-ROC."))
story.append(Image(str(FIG_DIR / "roc_curve.png"), width=12.2*cm, height=10.1*cm, hAlign="CENTER"))
story.append(Paragraph("Figura 2 - Curva ROC no conjunto de teste; AUC elevada indica separabilidade forte entre as classes.", styles["Smallx"]))

story.append(p("5. Ajuste Dinâmico do Threshold e Análise Financeira", "H1x"))
story.append(p(f"Foi adotada uma função de custo simples e auditável: <b>Custo Total = FN x {money(r['fn_cost'])} + FP x {money(r['fp_cost'])}</b>. O custo de um FN representa uma falha grave perdida, com potencial para parada não planejada, manutenção emergencial e risco operacional. O custo de um FP representa um falso alarme, com inspeção, intervenção ou parada preventiva. Nesta simulação, FN vale 10 vezes FP."))
story.append(Image(str(FIG_DIR / "threshold_cost.png"), width=16.6*cm, height=9.2*cm))
story.append(Paragraph(f"Figura 3 - Custo total no conjunto de validação utilizado para selecionar o threshold; o valor selecionado foi {optimized['threshold']:.2f}.", styles["Smallx"]))

story.append(p(f"Aplicando o threshold selecionado ao teste final, o custo estimado passou de <b>{money(standard['total_cost'])}</b> no threshold 0,50 para <b>{money(optimized['total_cost'])}</b>. A economia foi de <b>{money(r['savings'])}</b>, equivalente a <b>{r['savings_pct']:.2f}%</b>. O modelo, contudo, passou a gerar uma decisão mais conservadora em relação a falsos alarmes: o número de FP caiu de {standard['fp']} para {optimized['fp']}, enquanto FN subiu de {standard['fn']} para {optimized['fn']}. Isso ilustra que threshold ótimo depende da função de perda da operação, e não de uma regra universal de 0,50."))

story.append(p("Observação metodológica", "H2x"))
story.append(p("Para evitar overfitting do threshold ao teste, a busca do ponto de menor custo foi feita no conjunto de validação. O conjunto de teste somente recebeu o threshold já escolhido e foi usado para estimar o desempenho final. Em um sistema industrial real, a função de custo deve ser revisada com dados históricos de manutenção, tempo de parada, SLA, criticidade do ativo e custo de intervenção." , "Smallx"))

story.append(PageBreak())
story.append(p("6. Recomendações MLOps para Produção", "H1x"))
recs = [
    ["Monitoramento de dados", "Acompanhar distribuição dos sensores, taxa de eventos ausentes, outliers, drift e prevalência da classe Falha. Alertar quando houver mudança significativa no perfil operacional."],
    ["Monitoramento do modelo", "Monitorar Recall, Precisão, F-beta, AUC-ROC e principalmente FN. Quando a verdade de campo chegar com atraso, comparar predições com registros de manutenção e inspeção."],
    ["Threshold", "Manter o threshold como parâmetro versionado e configurável. Alterações devem ser registradas como mudança de configuração/modelo e submetidas a teste offline antes do deploy."],
    ["Observabilidade", "Registrar timestamp, ativo, versão do modelo, versão dos dados, probabilidade estimada, threshold aplicado e decisão final para auditoria e análise de incidentes."],
    ["Retreinamento", "Definir gatilhos por drift, degradação de métricas ou mudança do processo industrial. Reexecutar pipeline em ambiente reproduzível e manter artefatos versionados."],
    ["Governança", "Separar treino, validação e teste; controlar sementes aleatórias; versionar código, dependências e datasets; e utilizar CI/CD com testes de qualidade antes de promover o modelo."],
]
rt = Table([["Área", "Prática recomendada"]] + recs, colWidths=[4.0*cm, 12.5*cm], repeatRows=1)
rt.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16324F")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
    ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#C1CCD4")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F6F8FA")]),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("FONTSIZE", (0,0), (-1,-1), 8.2),
    ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
]))
story.append(rt)

story.append(p("7. Conclusão Executiva", "H1x"))
story.append(p(f"O experimento demonstra que, em cenários industriais fortemente desbalanceados, a acurácia não é uma métrica suficiente para orientar uma decisão de produção. A separação entre treino, validação e teste protege contra leakage e torna o experimento reprodutível. O modelo apresentou AUC-ROC de {standard['auc_roc']:.4f}, indicando forte capacidade discriminativa; entretanto, a escolha de threshold alterou significativamente o equilíbrio entre falsos positivos e falsos negativos. Sob a função de custo adotada, o threshold {optimized['threshold']:.2f} reduziu o custo do teste em {money(r['savings'])}, demonstrando o valor de conectar MLOps, métricas e economia operacional em uma mesma camada decisória."))
story.append(p("A entrega acompanha o código-fonte, as dependências, os dados sintéticos, os gráficos e os resultados numéricos. O projeto pode ser reexecutado no VSCode a partir do `requirements.txt` e do script `src/main.py`.", "Callout"))

report = Report(str(pdf_path), title="Relatório Executivo - MLOps Indústria 4.0", author="Projeto Prático")
report.build(story)
print(pdf_path)
