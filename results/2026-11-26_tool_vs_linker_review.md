# Second-rater review: 2026-11-26_tool_vs_linker.csv

Rater: Claude, blind to the key file. I labelled all 608 rows in `results/2026-11-26_tool_vs_linker_claude.csv`, using the rule as written and judging only the `evidence` line, before reading the author's verdicts.
Author's verdicts: right 111, borderline 144, wrong 353. Mine: right 141, borderline 184, wrong 283.

## 1. Agreement

`python -m esco_agent.labelsheet agree results/2026-11-26_tool_vs_linker.csv results/2026-11-26_tool_vs_linker_claude.csv`

```
387 suggestions labelled in both sheets

three verdicts
  agreement 0.81 [0.78, 0.85]   kappa 0.66 [0.59, 0.73] (chance 0.45)   AC1 0.74 [0.69, 0.80]

strict (right vs not)
  agreement 0.92 [0.89, 0.95]   kappa 0.70 [0.60, 0.80] (chance 0.73)   AC1 0.89 [0.85, 0.93]

lenient (wrong vs not)
  agreement 0.89 [0.86, 0.92]   kappa 0.77 [0.71, 0.84] (chance 0.52)   AC1 0.79 [0.73, 0.85]

confusion (rows: first sheet, columns: second)
                   right  borderline       wrong
right                 46           8           0
borderline            22          54           3
wrong                  1          38         215

72 disagreements:
  industrial-electronics: interpret circuit diagrams  (wrong / borderline)
  industrial-electronics: operate precision measuring equipment  (wrong / borderline)
  industrial-electronics: think analytically  (wrong / borderline)
  industrial-electronics: teach energy principles  (wrong / borderline)
  industrial-electronics: physics  (borderline / right)
  industrial-electronics: printed circuit boards  (borderline / wrong)
  industrial-electronics: teach electronics and automation principles  (wrong / borderline)
  industrial-electronics: calibrate laboratory equipment  (wrong / borderline)
  industrial-electronics: power electronics  (borderline / right)
  industrial-electronics: adjust voltage  (wrong / borderline)
  industrial-electronics: test instrumentation equipment  (wrong / borderline)
  industrial-electronics: teach electricity principles  (wrong / borderline)
  industrial-electronics: electricity  (borderline / right)
  industrial-electronics: test semiconductors  (wrong / borderline)
  industrial-electronics: build electronic prototypes  (right / borderline)
  industrial-electronics: design electronic systems  (borderline / right)
  industrial-electronics: using precision measuring equipment  (borderline / right)
  maths-and-statistics: teach basic numeracy skills  (wrong / borderline)
  maths-and-statistics: interpret visual literacy  (right / borderline)
  maths-and-statistics: execute analytical mathematical calculations  (right / borderline)
  maths-and-statistics: algebra  (borderline / right)
  maths-and-statistics: risk modelling  (wrong / borderline)
  maths-and-statistics: teach mathematics  (wrong / borderline)
  maths-and-statistics: advise on financial matters  (wrong / borderline)
  maths-and-statistics: create data sets  (wrong / right)
  maths-and-statistics: statistical analysis system software  (wrong / borderline)
  maths-and-statistics: scientific modelling  (wrong / borderline)
  maths-and-statistics: develop predictive models  (right / borderline)
  maths-and-statistics: process data  (wrong / borderline)
  maths-and-statistics: ability to understand and work with numbers  (borderline / right)
  maths-and-statistics: calculating and estimating  (borderline / right)
  maths-and-statistics: mathematics and statistics  (borderline / right)
  molecular-cell-biology-labs: biological chemistry  (wrong / borderline)
  molecular-cell-biology-labs: label medical laboratory samples  (wrong / borderline)
  molecular-cell-biology-labs: disseminate results to the scientific community  (right / borderline)
  molecular-cell-biology-labs: perform PCR covid test  (wrong / borderline)
  molecular-cell-biology-labs: diagnostic methods in medical laboratory  (wrong / borderline)
  molecular-cell-biology-labs: genomics  (wrong / borderline)
  molecular-cell-biology-labs: laboratory methods in biomedical sciences  (borderline / right)
  molecular-cell-biology-labs: assist in the production of laboratory documentation  (borderline / right)
  molecular-cell-biology-labs: conduct genome research  (wrong / borderline)
  molecular-cell-biology-labs: gather experimental data  (borderline / right)
  molecular-cell-biology-labs: molecular biology  (borderline / right)
  molecular-cell-biology-labs: biology  (borderline / right)
  molecular-cell-biology-labs: prepare scientific reports  (borderline / right)
  molecular-cell-biology-labs: analyse experimental laboratory data  (borderline / right)
  molecular-cell-biology-labs: report analysis results  (borderline / right)
  molecular-cell-biology-labs: evaluate genetic data  (wrong / borderline)
  molecular-cell-biology-labs: biochemistry  (borderline / right)
  plurilingual-intercultural-education: speak different languages  (wrong / borderline)
  plurilingual-intercultural-education: demonstrate intercultural competences in hospitality services  (wrong / borderline)
  plurilingual-intercultural-education: foreign languages for international careers  (wrong / borderline)
  plurilingual-intercultural-education: apply intercultural teaching strategies  (right / borderline)
  plurilingual-intercultural-education: perform manual work autonomously  (wrong / borderline)
  plurilingual-intercultural-education: handle tasks independently  (borderline / right)
  programming-fundamentals: Sass  (wrong / borderline)
  programming-fundamentals: object-oriented modelling  (borderline / wrong)
  programming-fundamentals: use scripting programming  (wrong / borderline)
  programming-fundamentals: web programming  (borderline / right)
  programming-fundamentals: maintain responsive design  (right / borderline)
  programming-fundamentals: task algorithmisation  (borderline / right)
  programming-fundamentals: data storage  (borderline / right)
  programming-fundamentals: design application interfaces  (borderline / wrong)
  tourism-economics: teach tourism principles  (wrong / borderline)
  tourism-economics: government policy  (right / borderline)
  tourism-economics: develop tourism products  (wrong / borderline)
  tourism-economics: educate on sustainable tourism  (wrong / borderline)
  tourism-economics: tourism sector policies  (wrong / borderline)
  tourism-economics: handle touristic quantitative data  (wrong / borderline)
  tourism-economics: assess an area as a tourism destination  (wrong / borderline)
  tourism-economics: measure sustainability of tourism activities  (wrong / borderline)
  tourism-economics: teach economic principles  (wrong / borderline)
```

**The tool compared only 387 of the 608 rows.** `agree` matches rows by `(course, skill_id)`. In this sheet the same skill appears on several lines of one course: 328 rows share their key with at least one other row. For each repeated key only the last row is kept, so 221 rows are left out. The output above therefore covers 64% of the sheet. The same row-level computation on all 608 rows, using the module's own `chance_corrected` and no bootstrap, gives:

| split | agreement | Cohen's kappa | Gwet's AC1 |
|---|---|---|---|
| three verdicts | 0.78 | 0.65 | 0.69 |
| strict (right vs not) | 0.91 | 0.72 | 0.86 |
| lenient (wrong vs not) | 0.87 | 0.75 | 0.74 |

Row-level confusion table (rows: author, columns: Claude):

| | right | borderline | wrong |
|---|---|---|---|
| right | 98 | 12 | 1 |
| borderline | 42 | 99 | 3 |
| wrong | 1 | 73 | 279 |

There are 132 disagreements, and 130 of them are one step apart. The author is stricter in a consistent direction: 73 rows the author marked wrong where I said borderline, and 42 the author marked borderline where I said right. The two raters almost never disagree on right vs. wrong (2 rows).

Disagreement rate by course: industrial-electronics 27/122 (0.22), maths-and-statistics 16/111 (0.14), molecular-cell-biology-labs 29/89 (0.33), plurilingual-intercultural-education 14/71 (0.20), programming-fundamentals 20/119 (0.17), tourism-economics 26/96 (0.27).

## 2. Every disagreement

Most disagreements fall into two groups:
- **17 rows** are teaching or training skills ("teach …", "provide training …", "educate on …"). The author marked them wrong; I marked them borderline because the rule's literal borderline case is "topic named, activity not shown".
- **29 rows** are knowledge skills on a line that names the topic (e.g. "web programming" on "3.1. HyperText Markup Language (HTML)"). The author marked them borderline; I marked them right.

Where a row ends "the author's … is defensible", I think the rule does not settle it.

| course | row | evidence | skill | author | Claude | why |
|---|---|---|---|---|---|---|
| industrial-electronics | 1 | 6 - Concept of Power and Energy, Joule's Law. | teach energy principles | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| industrial-electronics | 3 | Building electronic circuits. | electronics | borderline | right | Building circuits is the electronics topic in practice, so it is covered (right). |
| industrial-electronics | 8 | Using instrumentation and measurement equipment. | operate precision measuring equipment | wrong | borderline | ESCO means gauging part sizes (calipers, micrometers); lab measuring is only partly that. The author's wrong is defensible. |
| industrial-electronics | 9 | 12 - Semiconductors; | design integrated circuits | wrong | borderline | 'Semiconductors' names IC material but not IC design: borderline at most. The author's wrong is defensible. |
| industrial-electronics | 17 | Building electronic circuits. | build electronic prototypes | right | borderline | 'Building electronic circuits' is close to building prototypes. The author's right is defensible. |
| industrial-electronics | 18 | 8 - Introduction to alternate current, frequency and period; | electricity | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| industrial-electronics | 19 | Using instrumentation and measurement equipment. | test instrumentation equipment | wrong | borderline | Students use instruments but are not shown testing them: partial (borderline). The author's wrong is defensible. |
| industrial-electronics | 21 | Using instrumentation and measurement equipment. | using precision measuring equipment | borderline | right | The label means 'using measuring equipment' and the line says exactly that, so right. |
| industrial-electronics | 26 | 6 - Concept of Power and Energy, Joule's Law. | physics | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| industrial-electronics | 28 | 8 - Introduction to alternate current, frequency and period; | teach electricity principles | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| industrial-electronics | 30 | 14 - Introduction to power supplies. | power electronics | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| industrial-electronics | 41 | Perform accurate measurements with appropriate laboratory instruments. | operate precision measuring equipment | wrong | borderline | Same as row 8: ESCO's part-gauging sense only partly matches lab measurement. The author's wrong is defensible. |
| industrial-electronics | 47 | Undertake analysis and testing of electronic circuits. | electronics principles | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| industrial-electronics | 52 | Know the passive and active components for electronic circuits. | printed circuit boards | borderline | wrong | PCBs are neither named nor covered by 'passive and active components': wrong. |
| industrial-electronics | 55 | Acquire skills to design electronic circuits. | design electronic systems | borderline | right | The line shows designing electronic circuits; CAD appears only in the definition, not the label: right. |
| industrial-electronics | 56 | Undertake analysis and testing of electronic circuits. | electronics | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| industrial-electronics | 62 | 12 - Semiconductors; | test semiconductors | wrong | borderline | 'Semiconductors' is named but testing is not shown: borderline. The author's wrong is defensible. |
| industrial-electronics | 67 | Acquire skills to design electronic circuits. | design integrated circuits | wrong | borderline | Designing circuits is shown but not ICs: partial (borderline). |
| industrial-electronics | 71 | 14 - Introduction to power supplies. | electricity | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| industrial-electronics | 91 | Recognize the basic principles of electronics. | teach electronics and automation principles | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| industrial-electronics | 94 | Know the passive and active components for electronic circuits. | electronics principles | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| industrial-electronics | 99 | 8 - Introduction to alternate current, frequency and period; | electricity principles | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| industrial-electronics | 100 | Perform accurate measurements with appropriate laboratory instruments. | calibrate laboratory equipment | wrong | borderline | Measuring is shown but calibration is not: borderline. The author's wrong is defensible. |
| industrial-electronics | 105 | 5 - Voltage divider, current divider. | adjust voltage | wrong | borderline | A voltage divider sets a voltage, close to 'adjust voltage', but not in equipment: borderline. |
| industrial-electronics | 107 | 7 - nodal analysis, mesh analysis. | think analytically | wrong | borderline | Nodal analysis is analytical, but the skill is too generic to be sure: borderline. The author's wrong is defensible (different sense of 'analysis'). |
| industrial-electronics | 113 | Association of resistors in series and parallel; | electricity principles | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| industrial-electronics | 118 | Association of resistors in series and parallel; | interpret circuit diagrams | wrong | borderline | Series/parallel resistor networks are read from circuit diagrams, but that is not shown: borderline. |
| maths-and-statistics | 12 | 4. Organization and Description of Data. / Types of variables and scales of measures | process data | wrong | borderline | ESCO means clerical data entry; 'organization of data' is only partly that: borderline. The author's wrong is defensible. |
| maths-and-statistics | 30 | Distribution and parameters of a random variable. | mathematics and statistics | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| maths-and-statistics | 46 | Apply the functions in modeling real hypotheses. | scientific modelling | wrong | borderline | Modelling hypotheses with functions is partly scientific modelling: borderline. |
| maths-and-statistics | 57 | The curricular unit aims to work on basic concepts in specific areas of mathematics, namely Calculus and Probability and Statistics. | teach mathematics | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| maths-and-statistics | 58 | Application in the resolution of financial problems. | advise on financial matters | wrong | borderline | The line names financial problems but no advising: borderline. The author's wrong is defensible. |
| maths-and-statistics | 63 | Collect, organize and define a set of data through exploratory data analysis techniques. | create data sets | wrong | right | The line shows students collecting, organising and defining a set of data, which is the skill: right. |
| maths-and-statistics | 77 | Mathematically formulate simple problems arising from the performance of multiple functions in the management area. | carry out calculations | right | borderline | Formulating problems is not carrying out calculations: borderline. |
| maths-and-statistics | 79 | Polynomial and rational functions. | algebra | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| maths-and-statistics | 81 | Mathematically formulate simple problems arising from the performance of multiple functions in the management area. | execute analytical mathematical calculations | right | borderline | Formulating problems is not carrying out calculations: borderline. |
| maths-and-statistics | 82 | Interval estimates for the mean, variance and proportions. | calculating and estimating | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| maths-and-statistics | 88 | Graphically interpret real functions of real variable. | interpret visual literacy | right | borderline | Graphs of functions are only one kind of the 'charts, maps, graphics' in the definition: borderline. The author's right is defensible. |
| maths-and-statistics | 89 | 2. Successions of Real Numbers. | ability to understand and work with numbers | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| maths-and-statistics | 94 | The curricular unit aims to work on basic concepts in specific areas of mathematics, namely Calculus and Probability and Statistics. | teach basic numeracy skills | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| maths-and-statistics | 96 | Apply the functions in modeling real hypotheses. | develop predictive models | right | borderline | Modelling hypotheses with functions is close to predictive models, but prediction is not shown: borderline. The author's right is defensible. |
| maths-and-statistics | 103 | Use software as a tool for data analysis. | statistical analysis system software | wrong | borderline | SAS is not named; 'software' is generic: borderline. The author's wrong is defensible. |
| maths-and-statistics | 109 | 5. Variables and Probability Distributions. | risk modelling | wrong | borderline | Probability distributions are part of risk modelling, but risk is not shown: borderline. The author's wrong is defensible. |
| molecular-cell-biology-labs | 2 | 03. Identify, prevent and discuss sources of error in experimental procedures. | apply safety procedures in laboratory | right | wrong | The line is about experimental error, not safety: wrong. |
| molecular-cell-biology-labs | 8 | C2. Preparation of the laboratory notebook in Biosciences: recording, traceability and good practices (O4, O5, O6). | laboratory methods in biomedical sciences | wrong | borderline | A lab notebook is a lab practice but not biomedical methods: borderline. The author's wrong is defensible. |
| molecular-cell-biology-labs | 10 | 02. Plan and execute, with rigour and safety, techniques for the extraction, quantification and analysis of proteins and nucleic acids. | biological chemistry | wrong | borderline | The ESCO sense is a medical specialty, but the line is biochemistry lab work: borderline. The author's wrong is defensible. |
| molecular-cell-biology-labs | 15 | C6. Spectrophotometric and colorimetric methods for biomolecule quantification (O2, O3, O4). | biochemistry | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| molecular-cell-biology-labs | 20 | 03. Identify, prevent and discuss sources of error in experimental procedures. | analyse experimental laboratory data | wrong | borderline | Discussing error sources is part of analysing data, but analysis is not shown: borderline. |
| molecular-cell-biology-labs | 22 | C8. Conventional PCR and RT-PCR: | genomics | wrong | borderline | PCR is a genetics technique, not whole-genome study: borderline at most. The author's wrong is defensible. |
| molecular-cell-biology-labs | 24 | C1. Review of basic concepts in cell and molecular biology (O1). | clinical cytology | right | borderline | 'Clinical' does not fit, but the definition (cell structure and function) is covered: borderline. The author's right is defensible. |
| molecular-cell-biology-labs | 27 | 03. Identify, prevent and discuss sources of error in experimental procedures. | gather experimental data | wrong | borderline | Discussing error sources touches data gathering, but gathering is not shown: borderline. The author's wrong is defensible. |
| molecular-cell-biology-labs | 31 | C5. Extraction of nucleic acids (DNA and RNA) from biological samples (O2, O3, O4). | genomics | wrong | borderline | DNA extraction feeds genomics but does not cover it: borderline at most. The author's wrong is defensible. |
| molecular-cell-biology-labs | 32 | 01. Understand the fundamentals of cell and molecular biology laboratory techniques and their operating principles. | biology | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| molecular-cell-biology-labs | 34 | 04. Analyse, synthesise and discuss experimental results, communicating them in writing in a scientific and structured manner. | report the results of treatment | right | borderline | The definition (write up results) fits, but the 'treatment' context does not: borderline. The author's right is defensible. |
| molecular-cell-biology-labs | 41 | C1. Review of basic concepts in cell and molecular biology (O1). | biology | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| molecular-cell-biology-labs | 47 | C8. Conventional PCR and RT-PCR: | perform PCR covid test | wrong | borderline | RT-PCR is named but clinical covid testing is not shown: borderline. The author's wrong is defensible. |
| molecular-cell-biology-labs | 48 | C6. Spectrophotometric and colorimetric methods for biomolecule quantification (O2, O3, O4). | diagnostic methods in medical laboratory | wrong | borderline | Spectrophotometry is used in diagnostics, but diagnosis is not shown: borderline. The author's wrong is defensible. |
| molecular-cell-biology-labs | 49 | genotypic typing by PCR-RFLP — principles, enzymatic digestion and fragment pattern analysis (O2, O3, O4). | conduct genome research | wrong | borderline | Genotyping by PCR-RFLP is partly genome research: borderline. |
| molecular-cell-biology-labs | 52 | C6. Spectrophotometric and colorimetric methods for biomolecule quantification (O2, O3, O4). | biological chemistry | wrong | borderline | The ESCO sense is a medical specialty, but the line is biochemistry lab work: borderline. The author's wrong is defensible. |
| molecular-cell-biology-labs | 55 | 05. Organise a clear, complete and traceable laboratory notebook. | assist in the production of laboratory documentation | borderline | right | Organising a traceable lab notebook is documenting lab work: right. |
| molecular-cell-biology-labs | 62 | experimental design, execution and interpretation; | analyse experimental laboratory data | borderline | right | 'Execution and interpretation' of experiments is analysing lab data: right. |
| molecular-cell-biology-labs | 65 | 05. Organise a clear, complete and traceable laboratory notebook. | label medical laboratory samples | wrong | borderline | Traceability in a notebook is not labelling samples: borderline. The author's wrong is defensible. |
| molecular-cell-biology-labs | 66 | genotypic typing by PCR-RFLP — principles, enzymatic digestion and fragment pattern analysis (O2, O3, O4). | evaluate genetic data | wrong | borderline | Analysing fragment patterns is evaluating genetic data, but without statistics: borderline. |
| molecular-cell-biology-labs | 70 | 01. Understand the fundamentals of cell and molecular biology laboratory techniques and their operating principles. | laboratory methods in biomedical sciences | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| molecular-cell-biology-labs | 73 | C4. Extraction of proteins from biological material (O2, O3, O4). | biological chemistry | wrong | borderline | The ESCO sense is a medical specialty, but the line is biochemistry lab work: borderline. The author's wrong is defensible. |
| molecular-cell-biology-labs | 76 | experimental design, execution and interpretation; | gather experimental data | borderline | right | 'Experimental design, execution' covers gathering experimental data: right. |
| molecular-cell-biology-labs | 77 | genotypic typing by PCR-RFLP — principles, enzymatic digestion and fragment pattern analysis (O2, O3, O4). | genomics | wrong | borderline | PCR-RFLP genotyping is genetics, not genomics: borderline at most. The author's wrong is defensible. |
| molecular-cell-biology-labs | 78 | C2. Preparation of the laboratory notebook in Biosciences: recording, traceability and good practices (O4, O5, O6). | assist in the production of laboratory documentation | borderline | right | Preparing a lab notebook with traceability is documenting lab work: right. |
| molecular-cell-biology-labs | 80 | C3. Organisation of scientific reports: structure, critical analysis and discussion of results (O3, O4, O5, O7). | report analysis results | borderline | right | Organising reports with discussion of results is reporting analysis results: right. |
| molecular-cell-biology-labs | 83 | 04. Analyse, synthesise and discuss experimental results, communicating them in writing in a scientific and structured manner. | disseminate results to the scientific community | right | borderline | The line shows written communication but not dissemination to the community: borderline. |
| molecular-cell-biology-labs | 86 | 01. Understand the fundamentals of cell and molecular biology laboratory techniques and their operating principles. | molecular biology | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| molecular-cell-biology-labs | 87 | C3. Organisation of scientific reports: structure, critical analysis and discussion of results (O3, O4, O5, O7). | prepare scientific reports | borderline | right | Organising scientific reports covers preparing them: right. |
| plurilingual-intercultural-education | 1 | - Debate different issues related to plurilingual and intercultural education; | teach intercultural communication methods | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| plurilingual-intercultural-education | 2 | The world of languages and the languages of the world. | foreign languages for international careers | wrong | borderline | Languages are the topic, but not as a career tool: borderline. The author's wrong is defensible. |
| plurilingual-intercultural-education | 3 | The world of languages and the languages of the world. | speak different languages | wrong | borderline | Languages are the topic, but speaking them is not shown: borderline. |
| plurilingual-intercultural-education | 12 | - Mobilise strategies for developing plurilingual and intercultural competence; | foreign languages for international careers | wrong | borderline | Plurilingual competence is related, but career languages are not shown: borderline. The author's wrong is defensible. |
| plurilingual-intercultural-education | 16 | - Be aware of the role of languages and language learning in personal and social development; | language acquisition | borderline | right | 'Language learning' is named directly: right. |
| plurilingual-intercultural-education | 19 | Plural approaches and language teaching in the first years of school. | study acquisition of language | wrong | borderline | Language teaching to young children touches acquisition, but its study is not shown: borderline. The author's wrong is defensible. |
| plurilingual-intercultural-education | 32 | - Discuss the educational implications of different types of language policies; | language teaching methods | wrong | borderline | Language policy is not teaching method: borderline at most. The author's wrong is defensible. |
| plurilingual-intercultural-education | 34 | - Learn about different approaches to language education; | study acquisition of language | wrong | borderline | Approaches to language education touch acquisition: borderline. The author's wrong is defensible. |
| plurilingual-intercultural-education | 38 | - Create and develop a taste for other languages and cultures as forms of communication and expression; | speak different languages | wrong | borderline | A taste for other languages is not speaking them: borderline. The author's wrong is defensible. |
| plurilingual-intercultural-education | 40 | - Demonstrate the ability to work autonomously and as part of a team; | perform manual work autonomously | wrong | borderline | Autonomy is shown but manual work is not: borderline. The author's wrong (different sense) is defensible. |
| plurilingual-intercultural-education | 42 | - Demonstrate the ability to work autonomously and as part of a team; | handle tasks independently | borderline | right | The line shows students working autonomously, which is the skill: right. |
| plurilingual-intercultural-education | 49 | The speaker and his/her languages: linguistic biographies and tools for recording experiences of contact with other languages and cultures. | modern languages | right | borderline | Linguistic biographies cover a speaker's languages, not modern languages as such: borderline. The author's right is defensible. |
| plurilingual-intercultural-education | 62 | - Mobilise strategies for developing plurilingual and intercultural competence; | demonstrate intercultural competences in hospitality services | wrong | borderline | Intercultural competence is shown but hospitality is not: borderline. The author's wrong is defensible. |
| plurilingual-intercultural-education | 64 | - Mobilise strategies for developing plurilingual and intercultural competence; | apply intercultural teaching strategies | right | borderline | The line develops the students' own competence, not their teaching: borderline. |
| programming-fundamentals | 2 | • Is able to develop a responsive web application of medium complexity | maintain responsive design | right | borderline | Students build a responsive app but do not maintain a site: borderline. The author's right is defensible. |
| programming-fundamentals | 3 | • Autonomously analyzes and processes programming concepts, models and techniques; | object-oriented modelling | borderline | wrong | 'Models' is a word match; OO modelling is not shown: wrong. The author's borderline is defensible. |
| programming-fundamentals | 9 | 3.1. HyperText Markup Language (HTML) | web programming | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| programming-fundamentals | 17 | 2.1. W3C Browsers and Standards | use markup languages | wrong | borderline | W3C standards cover HTML, but using markup is not shown: borderline. The author's wrong is defensible. |
| programming-fundamentals | 20 | • Identifies the current languages and technologies appropriate for web application development; | web programming | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| programming-fundamentals | 21 | 1.4. How to code an algorithm | use scripting programming | wrong | borderline | Coding an algorithm is close to scripting, but scripting is not named: borderline. The author's wrong is defensible. |
| programming-fundamentals | 27 | • Identifies the current languages and technologies appropriate for web application development; | ASP.NET | wrong | borderline | The line lists web technologies without naming ASP.NET: borderline. The author's wrong is defensible. |
| programming-fundamentals | 30 | 4. Web Application Development | web programming | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| programming-fundamentals | 32 | 3.3. Web programming languages | use markup languages | wrong | borderline | Web languages include markup, but using it is not shown: borderline. The author's wrong is defensible. |
| programming-fundamentals | 39 | 4.3. Storage layer | data storage | borderline | right | 'Storage layer' names data storage: right. |
| programming-fundamentals | 40 | • Recognizes and knows how to use the style language CSS, client language JavaScript, server lanaguages PHP and ASP.NET | use markup languages | wrong | borderline | CSS is a style-sheet language, not markup: borderline at most. The author's wrong is defensible. |
| programming-fundamentals | 42 | • Autonomously analyzes and processes programming concepts, models and techniques; | use scripting programming | wrong | borderline | Programming concepts are generic, scripting is not named: borderline. The author's wrong is defensible. |
| programming-fundamentals | 44 | 4.1. Front-End Development | design application interfaces | borderline | wrong | ESCO means APIs (software interfaces), the line is front-end UI: different sense. The author's borderline is defensible. |
| programming-fundamentals | 47 | • Knows and knows how to use web page markup language HTML; | web programming | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| programming-fundamentals | 64 | 3.4. Responsive web application development frameworks | web programming | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| programming-fundamentals | 66 | 3.2. Cascading Style Sheets (CSS) | Sass | wrong | borderline | Sass is a CSS preprocessor that is not named: borderline. The author's wrong is defensible. |
| programming-fundamentals | 67 | • Is able to develop a responsive web application of medium complexity | web programming | borderline | right | The line shows students developing a web application: right. |
| programming-fundamentals | 77 | 4.1. Front-End Development | web programming | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| programming-fundamentals | 83 | • Designs algorithms based on real situations of simple to moderate complexity, applying appropriate programming methodologies; | use software design patterns | wrong | borderline | 'Programming methodologies' is near design patterns, but they are not named: borderline. The author's wrong is defensible. |
| programming-fundamentals | 114 | 1. Introduction to Algorithm | task algorithmisation | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| tourism-economics | 1 | • Understand how tourism can contribute to the Sustainable Development Goals. | provide training in sustainable tourism development and management | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| tourism-economics | 3 | It is intended to provide students with basic competences that makes them more qualified to deal professionally with the new challenges and demands that tourism sector faces. | teach tourism principles | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| tourism-economics | 5 | • Identifify and interpret a set of indicators related to tourism economics. | assess an area as a tourism destination | wrong | borderline | Tourism indicators are partly a way to assess destinations: borderline. The author's wrong is defensible. |
| tourism-economics | 9 | 6. Tourism Satellite Account. | travel, tourism and leisure | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| tourism-economics | 14 | • Recognize the importance of innovation in tourism. | develop tourism products | wrong | borderline | Innovation in tourism is near product development, which is not shown: borderline. |
| tourism-economics | 16 | • Apply the key concepts and techniques of economics to the analysis of the tourism market. | report touristic facts | wrong | borderline | Analysing the tourism market is part of reporting tourism facts: borderline. |
| tourism-economics | 21 | 4. Tourist motivations | teach tourism principles | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| tourism-economics | 26 | • Identifify and interpret a set of indicators related to tourism economics. | measure sustainability of tourism activities | wrong | borderline | Interpreting indicators is close, but sustainability is not named: borderline. The author's wrong is defensible. |
| tourism-economics | 27 | 2. Theoretical framework and conceptualization of tourism. | teach tourism principles | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| tourism-economics | 28 | • Recognize tourism as one of the key sectors of development in all countries and a major source of income, jobs and wealth creation. | develop tourism policies | wrong | borderline | The sector's importance is named, policy is not: borderline. The author's wrong is defensible. |
| tourism-economics | 33 | Introdução à Economia do Turismo aims at offering the students their first contact with the basic concepts of the economics of tourism. More specifically , the purpose of this course is to apply basic microeconomics and macroeconomics concepts to tourism. | tourism market | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| tourism-economics | 35 | 5.1 Economic analysis of tourism demand and supply | handle touristic quantitative data | wrong | borderline | Demand and supply analysis uses quantitative data, which is not shown: borderline. |
| tourism-economics | 38 | 7. The economic impact of tourism. | travel, tourism and leisure | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| tourism-economics | 39 | 8. Other topics of tourism economics- sustainability and innovation in tourism. | provide training in sustainable tourism development and management | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| tourism-economics | 40 | Introdução à Economia do Turismo aims at offering the students their first contact with the basic concepts of the economics of tourism. More specifically , the purpose of this course is to apply basic microeconomics and macroeconomics concepts to tourism. | teach tourism principles | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| tourism-economics | 41 | 1. Basic principles of economic analysis | teach economic principles | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| tourism-economics | 46 | • Understand how tourism can contribute to the Sustainable Development Goals. | educate on sustainable tourism | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| tourism-economics | 48 | • Identifify and interpret a set of indicators related to tourism economics. | report touristic facts | wrong | borderline | Interpreting tourism indicators is part of reporting tourism facts: borderline. |
| tourism-economics | 51 | • Understand how tourism can contribute to the Sustainable Development Goals. | measure sustainability of tourism activities | wrong | borderline | The topic (tourism and the SDGs) is named, measuring is not: borderline. On the same line the author marked 'ecotourism' borderline. |
| tourism-economics | 54 | 5.4 Governmental interventation | government policy | right | borderline | Government intervention in markets is government policy in an economics sense: borderline. The author's right is defensible. |
| tourism-economics | 69 | 4. Tourist motivations | travel, tourism and leisure | borderline | right | Knowledge: the line names or covers the topic, so the rule gives right; the author's borderline reads the heading as only partial coverage. |
| tourism-economics | 71 | 8. Other topics of tourism economics- sustainability and innovation in tourism. | educate on sustainable tourism | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| tourism-economics | 73 | Introdução à Economia do Turismo aims at offering the students their first contact with the basic concepts of the economics of tourism. More specifically , the purpose of this course is to apply basic microeconomics and macroeconomics concepts to tourism. | teach economic principles | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| tourism-economics | 92 | • Understand various concepts relevant in the context of tourism. | teach tourism principles | wrong | borderline | Teaching/training skill: topic named, students not shown teaching, which the rule calls borderline; the author's wrong holds if 'students never teach' counts as 'no support at all'. |
| tourism-economics | 94 | 5. The tourism market | tourism sector policies | wrong | borderline | The tourism market is named, sector policy is not: borderline. The author's wrong is defensible. |
| tourism-economics | 96 | 8. Other topics of tourism economics- sustainability and innovation in tourism. | measure sustainability of tourism activities | wrong | borderline | The topic (sustainability in tourism) is named, measuring is not: borderline. On the same line the author marked 'ecotourism' borderline. |

## 3. Rows clearly mislabelled under the rule

These are the rows where I think the author's verdict is outside what the rule allows. They are listed separately from the judgement calls above.

| course | row | evidence | skill | author | should be |
|---|---|---|---|---|---|
| molecular-cell-biology-labs | 2 | 03. Identify, prevent and discuss sources of error in experimental procedures. | apply safety procedures in laboratory | right | wrong (error ≠ safety; row 1's "with rigour and safety" line would fit, this one does not) |
| maths-and-statistics | 63 | Collect, organize and define a set of data through exploratory data analysis techniques. | create data sets | wrong | right (activity shown) |
| programming-fundamentals | 67 | • Is able to develop a responsive web application of medium complexity | web programming | borderline | right (activity shown) |
| programming-fundamentals | 47 | • Knows and knows how to use web page markup language HTML; | web programming | borderline | right |
| plurilingual-intercultural-education | 16 | - Be aware of the role of languages and language learning in personal and social development; | language acquisition | borderline | right (topic named) |
| molecular-cell-biology-labs | 55 | 05. Organise a clear, complete and traceable laboratory notebook. | assist in the production of laboratory documentation | borderline | right |
| molecular-cell-biology-labs | 62 | experimental design, execution and interpretation; | analyse experimental laboratory data | borderline | right |
| tourism-economics | 51, 96 | • Understand how tourism can contribute to the Sustainable Development Goals. / 8. Other topics of tourism economics- sustainability and innovation in tourism. | measure sustainability of tourism activities | wrong | borderline (topic named; see §4) |
| tourism-economics | 46, 71 | same two lines | educate on sustainable tourism | wrong | borderline (topic named) |

Everything else in §2 is a borderline call that either reading of the rule can support.

## 4. Consistency

The same skill is labelled differently on near-identical lines, or the same kind of case is labelled differently. All verdicts below are the author's.

- **web programming** (programming-fundamentals): right on "3.3. Web programming languages", "3. Introduction to Web Programming" and "• Recognizes and knows how to use the style language CSS, …". Borderline on "3.1. HyperText Markup Language (HTML)", "4. Web Application Development", "4.1. Front-End Development", "3.4. Responsive web application development frameworks" and "• Is able to develop a responsive web application of medium complexity". These are the same kind of line, and the last one shows the activity.
- **electricity principles** (industrial-electronics): right on "2 - Ohm's Law;", "5 - Voltage divider, current divider." and "3 - The power voltage and current, …". Borderline on "8 - Introduction to alternate current, frequency and period;" and "Association of resistors in series and parallel;".
- **electronics / electronics principles**: right on "Learn the fundamental concepts of electronics." and "1 - Basic concepts of electronics;". Borderline on "Undertake analysis and testing of electronic circuits." and "Know the passive and active components for electronic circuits.".
- **travel, tourism and leisure** (tourism-economics): right on 5 lines. Borderline on "6. Tourism Satellite Account.", "7. The economic impact of tourism." and "4. Tourist motivations". Every one of these lines is about tourism.
- **molecular biology**: right on "C1. Review of basic concepts in cell and molecular biology (O1).", borderline on "01. Understand the fundamentals of cell and molecular biology laboratory techniques …".
- **prepare scientific reports**: right on "O7. Produce a structured scientific report …", borderline on "C3. Organisation of scientific reports: …".
- **disseminate results to the scientific community**: right on "04. Analyse, synthesise and discuss experimental results, communicating them in writing …", but borderline on "C3. Organisation of scientific reports …". Neither line mentions dissemination, so these are lenient in the opposite direction from the cases above.
- **Sustainability in tourism**: same lines, different verdicts. On "8. Other topics of tourism economics- sustainability and innovation in tourism." and "• Understand how tourism can contribute to the Sustainable Development Goals.", *ecotourism* is borderline, but *measure sustainability of tourism activities*, *educate on sustainable tourism* and *provide training in sustainable tourism development and management* are wrong.
- **Teaching skills across courses.** Every "teach …" row outside plurilingual-intercultural-education is wrong (17 rows). Inside it, "teach languages" is borderline where the line names language teaching. This looks like a deliberate unwritten rule: a teaching skill counts only in a course that trains teachers. It is applied consistently, but the written rule doesn't say it. Under the literal rule these are "topic named, activity not shown" = borderline. This is the largest single source of disagreement, so the rule should state which reading applies.
- **"Topic named, activity not shown" for operating skills.** These are mostly treated consistently: "operate electronic measuring instruments" and "measure electrical characteristics" on the voltmeter line are borderline, and "operate signal generator" on the function generator line is borderline. But "test semiconductors" and "design integrated circuits" on "12 - Semiconductors;" are wrong. If those count as different senses (manufacturing vs. course content), the author's verdicts are consistent.
- No row has the same skill on the same line twice with two verdicts. The sheet has no exact duplicate rows.

## 5. Signs of labelling fatigue

- **Disagreement by position:** first third of each course 47/200 (0.23), middle 44/204 (0.22), last 41/204 (0.20). There is no overall rise in later rows. Per course it is mixed: maths-and-statistics rises (2/37 → 4/37 → 10/37) and molecular-cell-biology-labs rises slightly (8/29 → 9/30 → 12/30). Programming-fundamentals falls (10/39 → 8/40 → 2/40).
- **Long runs:** the longest runs of one verdict are "wrong" × 11 (maths-and-statistics rows 2–12) and "wrong" × 10 (molecular-cell-biology-labs rows 44–53). With about 58% "wrong" and around 100 rows per course, the expected longest run is about 7, so these are long but not extreme. I had 10 wrong in a row in the same maths stretch, so that run reflects the data. The molecular run includes 4 rows I marked borderline (rows 47 "perform PCR covid test", 48 "diagnostic methods in medical laboratory", 49 "conduct genome research", 52 "biological chemistry"). That is the one place that looks like a run of quick "wrong" verdicts.
- **Notes:** the author's `note` column is empty on all 608 rows. The rule asks for notes where the rater was unsure, so either the author was never unsure or notes were skipped.
- Overall I see no strong sign of fatigue. The main difference between the raters is systematic: the author is stricter throughout.

## 6. Other problems with the sheet

- **`agree` drops repeated skills** (see §1). The key `(course, skill_id)` is not unique in this sheet. Keying on `(course, row)` or `(course, skill_id, evidence)` would compare all 608 rows. I did not change the code.
- No empty evidence. Row numbers run 1..n in every course without gaps.
- **Evidence fragments:** some lines lose their context when split. "diode and the transistor." and "Its association in series and parallel." start mid-sentence, and "C8. Conventional PCR and RT-PCR:" ends at a colon, so its sub-items are on other lines. Judging "only the line shown" is harder on these rows.
- **Heavily repeated skills:** "tourism market" appears on 15 of the 20 tourism lines, "web programming" on 10 and "travel, tourism and leisure" on 8. These repeats drive both the `agree` key problem and most of the consistency issues in §4.
- **Definitions that differ from the label's everyday sense:** "protein" is defined as a nutrient (dietary sense), and "biological chemistry" as a medical specialty. ABAP, APL, ASP.NET, Assembly, Java and PHP all share the same generic definition ("The techniques and principles of software development …"), so the label is the only clue to which language is meant.
- **Non-English labels:** 22 rows have "(not an English ESCO skill label)" and an empty `reuse_level`. They are all broad labels such as "travel, tourism and leisure", "programming" and "law". They are easy to mark right whenever the topic appears, which affects precision.
- Typos carried from the source text: "Identifify", "interventation", "lanaguages", "Knows and knows how to use".
