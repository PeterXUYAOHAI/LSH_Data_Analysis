
SELECT d.author_id, doc_id, ct.para_id, ct.f1, ct.f2, ct.f3, ct.f4, ct.f5, ct.f6, ct.f7
, ct.f8, ct.f9, ct.f10, ct.f11, ct.f12, ct.f13, ct.f14, ct.f15, ct.f16, ct.f17, ct.f18, ct.f19
, ct.f20, ct.f21, ct.f22, ct.f23, ct.f24, ct.f25, ct.f26, ct.f27, ct.f28, ct.f29, ct.f30,
 ct.f31, ct.f32, ct.f33, ct.f34, ct.f35, ct.f36, ct.f37, ct.f38, ct.f39, ct.f40, ct.f41,
 ct.f42, ct.f43, ct.f44, ct.f45, ct.f46, ct.f47, ct.f48, ct.f49, ct.f50, ct.f51, ct.f52,
 ct.f53, ct.f54, ct.f55, ct.f56, ct.f57  FROM crosstab('SELECT para_id, 
feature_id ,feature_value FROM fact where doc_id between 1 and 20 ORDER BY para_id, feature_id', 'SELECT 
DISTINCT feature_id FROM fact WHERE doc_id between 1 and 20 ORDER BY feature_id') as ct(para_id int, f1 
real, f2 real, f3 real, f4 real, f5 real, f6 real, f7 real, f8 real, f9 real, f10 real, f11 
real, f12 real, f13 real, f14 real, f15 real, f16 real, f17 real, f18 real, f19 real, f20 
real, f21 real, f22 real, f23 real, f24 real, f25 real, f26 real, f27 real, f28 real, f29 
real, f30 real, f31 real, f32 real, f33 real, f34 real, f35 real, f36 real, f37 real, f38 
real, f39 real, f40 real, f41 real, f42 real, f43 real, f44 real, f45 real, f46 real, f47 
real, f48 real, f49 real, f50 real, f51 real, f52 real, f53 real, f54 real, f55 real, f56 
real, f57 real) INNER JOIN document d ON d.doc_id=doc_id;


\copy (select * from fact limit 10) TO 'C:/Users/yaoha_000/Desktop/testoutput.csv'  WITH (FORMAT CSV);
 
\copy (SELECT d.author_id, doc_id, ct.para_id, ct.f1, ct.f2, ct.f3, ct.f4, ct.f5, ct.f6, ct.f7, ct.f8, ct.f9, ct.f10, ct.f11, ct.f12, ct.f13, ct.f14, ct.f15, ct.f16, ct.f17, ct.f18, ct.f19, ct.f20, ct.f21, ct.f22, ct.f23, ct.f24, ct.f25, ct.f26, ct.f27, ct.f28, ct.f29, ct.f30,ct.f31, ct.f32, ct.f33, ct.f34, ct.f35, ct.f36, ct.f37, ct.f38, ct.f39, ct.f40, ct.f41,ct.f42, ct.f43, ct.f44, ct.f45, ct.f46, ct.f47, ct.f48, ct.f49, ct.f50, ct.f51, ct.f52,ct.f53, ct.f54, ct.f55, ct.f56, ct.f57  FROM crosstab('SELECT para_id, feature_id ,feature_value FROM fact where doc_id between 1 and 20 ORDER BY para_id, feature_id', 'SELECT DISTINCT feature_id FROM fact WHERE doc_id between 1 and 20 ORDER BY feature_id') as ct(para_id int, f1 real, f2 real, f3 real, f4 real, f5 real, f6 real, f7 real, f8 real, f9 real, f10 real, f11 real, f12 real, f13 real, f14 real, f15 real, f16 real, f17 real, f18 real, f19 real, f20 real, f21 real, f22 real, f23 real, f24 real, f25 real, f26 real, f27 real, f28 real, f29 real, f30 real, f31 real, f32 real, f33 real, f34 real, f35 real, f36 real, f37 real, f38 real, f39 real, f40 real, f41 real, f42 real, f43 real, f44 real, f45 real, f46 real, f47 real, f48 real, f49 real, f50 real, f51 real, f52 real, f53 real, f54 real, f55 real, f56 real, f57 real) INNER JOIN document d ON d.doc_id=doc_id) TO 'C:/Users/yaoha_000/Desktop/testoutput.csv'  WITH (FORMAT CSV);

\copy (select  d.doc_id, d.author_id ,ct.para_id, ct.f1, ct.f2, ct.f3, ct.f4, ct.f5, ct.f6, ct.f7, ct.f8, ct.f9, ct.f10, ct.f11, ct.f12, ct.f13, ct.f14, ct.f15, ct.f16, ct.f17, ct.f18, ct.f19, ct.f20, ct.f21, ct.f22, ct.f23, ct.f24, ct.f25, ct.f26, ct.f27, ct.f28, ct.f29, ct.f30, ct.f31, ct.f32, ct.f33, ct.f34, ct.f35, ct.f36, ct.f37, ct.f38, ct.f39, ct.f40, ct.f41, ct.f42, ct.f43, ct.f44, ct.f45, ct.f46, ct.f47, ct.f48, ct.f49, ct.f50, ct.f51, ct.f52, ct.f53, ct.f54, ct.f55, ct.f56, ct.f57  from crosstab('SELECT para_id, feature_id ,feature_value FROM fact  ORDER BY para_id, feature_id', 'SELECT DISTINCT feature_id FROM fact  ORDER BY feature_id') as ct(para_id int, f1 real, f2 real, f3 real, f4 real, f5 real, f6 real, f7 real, f8 real, f9 real, f10 real, f11 real, f12 real, f13 real, f14 real, f15 real, f16 real, f17 real, f18 real, f19 real, f20 real, f21 real, f22 real, f23 real, f24 real, f25 real, f26 real, f27 real, f28 real, f29 real, f30 real, f31 real, f32 real, f33 real, f34 real, f35 real, f36 real, f37 real, f38 real, f39 real, f40 real, f41 real, f42 real, f43 real, f44 real, f45 real, f46 real, f47 real, f48 real, f49 real, f50 real, f51 real, f52 real, f53 real, f54 real, f55 real, f56 real, f57 real) , (select doc_id, para_id from fact group by doc_id, para_id ) f , document d where ct.para_id = f.para_id and d.doc_id = f.doc_id) TO 'C:/Users/yaoha_000/Desktop/testoutput.csv'  WITH (FORMAT CSV);


select  d.doc_id, d.author_id ,ct.para_id, ct.f1, ct.f2, ct.f3, ct.f4, ct.f5, ct.f6, ct.f7
, ct.f8, ct.f9, ct.f10, ct.f11, ct.f12, ct.f13, ct.f14, ct.f15, ct.f16, ct.f17, ct.f18, ct.f19
, ct.f20, ct.f21, ct.f22, ct.f23, ct.f24, ct.f25, ct.f26, ct.f27, ct.f28, ct.f29, ct.f30,
 ct.f31, ct.f32, ct.f33, ct.f34, ct.f35, ct.f36, ct.f37, ct.f38, ct.f39, ct.f40, ct.f41,
 ct.f42, ct.f43, ct.f44, ct.f45, ct.f46, ct.f47, ct.f48, ct.f49, ct.f50, ct.f51, ct.f52,
 ct.f53, ct.f54, ct.f55, ct.f56, ct.f57  from 
crosstab('SELECT para_id, 
feature_id ,feature_value FROM fact where doc_id between 1 and 20 ORDER BY para_id, feature_id', 'SELECT 
DISTINCT feature_id FROM fact WHERE doc_id between 1 and 20 ORDER BY feature_id') as ct(para_id int, f1 
real, f2 real, f3 real, f4 real, f5 real, f6 real, f7 real, f8 real, f9 real, f10 real, f11 
real, f12 real, f13 real, f14 real, f15 real, f16 real, f17 real, f18 real, f19 real, f20 
real, f21 real, f22 real, f23 real, f24 real, f25 real, f26 real, f27 real, f28 real, f29 
real, f30 real, f31 real, f32 real, f33 real, f34 real, f35 real, f36 real, f37 real, f38 
real, f39 real, f40 real, f41 real, f42 real, f43 real, f44 real, f45 real, f46 real, f47 
real, f48 real, f49 real, f50 real, f51 real, f52 real, f53 real, f54 real, f55 real, f56 
real, f57 real) , (select doc_id, para_id from fact group by doc_id, para_id ) f , document d
where ct.para_id = f.para_id and 
d.doc_id = f.doc_id;


SELECT ct.para_id, ct.f1, ct.f2, ct.f3, ct.f4, ct.f5, ct.f6, ct.f7
, ct.f8, ct.f9, ct.f10, ct.f11, ct.f12, ct.f13, ct.f14, ct.f15, ct.f16, ct.f17, ct.f18, ct.f19
, ct.f20, ct.f21, ct.f22, ct.f23, ct.f24, ct.f25, ct.f26, ct.f27, ct.f28, ct.f29, ct.f30,
 ct.f31, ct.f32, ct.f33, ct.f34, ct.f35, ct.f36, ct.f37, ct.f38, ct.f39, ct.f40, ct.f41,
 ct.f42, ct.f43, ct.f44, ct.f45, ct.f46, ct.f47, ct.f48, ct.f49, ct.f50, ct.f51, ct.f52,
 ct.f53, ct.f54, ct.f55, ct.f56, ct.f57  FROM crosstab('SELECT para_id,
feature_id ,feature_value FROM fact  ORDER BY para_id, feature_id', 'SELECT 
DISTINCT feature_id FROM fact  ORDER BY feature_id') as ct(para_id int, f1 
real, f2 real, f3 real, f4 real, f5 real, f6 real, f7 real, f8 real, f9 real, f10 real, f11 
real, f12 real, f13 real, f14 real, f15 real, f16 real, f17 real, f18 real, f19 real, f20 
real, f21 real, f22 real, f23 real, f24 real, f25 real, f26 real, f27 real, f28 real, f29 
real, f30 real, f31 real, f32 real, f33 real, f34 real, f35 real, f36 real, f37 real, f38 
real, f39 real, f40 real, f41 real, f42 real, f43 real, f44 real, f45 real, f46 real, f47 
real, f48 real, f49 real, f50 real, f51 real, f52 real, f53 real, f54 real, f55 real, f56 
real, f57 real) limit 10;