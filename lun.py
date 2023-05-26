import pymongo
client=pymongo.MongoClient('mongodb://localhost:27017/?directConnection=true')
my=client['DB']
info=my.tab
res=[{
    'id':103,
    'name':'a'
},
{
    'id':104,
    'name':'ab'

},{
    'id':105,
    'name':'bhay'

}]
# info.insert_many(res)
# z=info.find_one({},{'_id':0})

z=info.find({},{'_id':0})
for document in z:
    # Process each document as needed
    print(document)
