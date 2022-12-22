# ML_book_recommendation
Book Rental Recommendation
Before reading data from a csv file, you need to download the “BX-Book-Ratings.csv”, “BX-Books.csv”, “BX-Users.csv”, and “Recommend.csv” datasets from the resource section and upload them into the Lab. We will use the Up arrow icon, which is shown on the left side under the View icon. Click on the Up arrow icon and upload the file from wherever it was downloaded into your system.

The objective is to recommend books to a user based on their purchase history and the behavior of other users.

Dataset Description
BX-Users: It contains the information of users.
user_id - These have been anonymized and mapped to integers
Location - Demographic data is provided
Age - Demographic data is provided
If available. Otherwise, these fields contain NULL-values.

BX-Books:

isbn - Books are identified by their respective ISBNs. Invalid ISBNs have already been removed from the dataset.
book_title
book_author
year_of_publication
publisher
BX-Book-Ratings: Contains the book rating information.
user_id
isbn
rating - Ratings (Book-Rating) are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.
