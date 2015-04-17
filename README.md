# install spark first, and use command:

"spark-submit dsgd_mf.py num_factors num_workers num_iterations beta lambda V.csv w.csv h.csv"

where V.csv is the input matrix csv file, with each line : user_id, movie_id, rate
w.csv h.csv are 2 output csv files
