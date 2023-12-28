import { useRouter } from 'next/router';
import ProductCard from '../../components/ProductCard';
import styles from '../../styles/ShopPage.module.css';
import { getProducts } from '../api/products/index';

const ShopPage = ({ products }) => {
  const router = useRouter();
  const searchTerm = router.query.search || '';

  const filteredProducts = products.filter(product =>
    product.product.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>All Results</h1>
      <div className={styles.cards}>
        {filteredProducts.length > 0 ? (
          filteredProducts.map((product) => (
            <ProductCard key={product.id} product={product} />
          ))
        ) : (
          <p>No products found</p>
        )}
      </div>
    </div>
  );
};

export default ShopPage;

export async function getStaticProps() {
  const products = await getProducts();
  return { props: { products } };
}