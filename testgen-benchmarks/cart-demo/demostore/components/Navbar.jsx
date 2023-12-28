import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { useSelector } from 'react-redux';
import styles from '../styles/Navbar.module.css';

const Navbar = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const router = useRouter();
  const cart = useSelector((state) => state.cart);

  const getItemsCount = () => {
    return "Cart (" + cart.reduce((accumulator, item) => accumulator + item.quantity, 0) + ")";
  };

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (searchTerm.trim()) {
      router.push(`/shop?search=${searchTerm}`);
    }
  };

  const userText = () => {
    return "User";
    if (router.pathname === '/shop') {
      return "User";
    } else {
      return "";
    }
  };

  return (
    <nav className={styles.navbar}>
      <h6 className={styles.logo}>ROBO_STORE</h6>
      <form onSubmit={handleSearchSubmit} className={styles.searchForm}>
        <input 
          type="text" 
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Search products"
          className={styles.searchInput}
        />
        <button type="submit" className={styles.searchButton}>Search</button>
      </form>
      <ul className={styles.links}>
        <li className={styles.navlink}>
          <Link href="/">Home</Link>
        </li>
        <li className={styles.navlink}>
          <Link href="/shop">Shop</Link>
        </li>
        <li className={styles.navlink}>
          <Link href="/user">{userText()}</Link>
        </li>
        <li className={styles.navlink}>
          <Link href="/cart">
            <a>{getItemsCount()}</a>
          </Link>
        </li>
      </ul>
    </nav>
  );
};

export default Navbar;