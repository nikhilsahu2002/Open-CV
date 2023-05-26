import pymongo
client=pymongo.MongoClient('mongodb://localhost:27017/?directConnection=true')
my=client['DB']
info=my.tab
res=[{
    'id':102,
    'name':'abhay',
    'image':'imag1.png'

}]
info.insert_many(res)